import math
from re import T
import numpy as np
import pdb
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List
from torch.profiler import profile, record_function, ProfilerActivity
from collections import defaultdict

# from gpu_mem_track import MemTracker
from einops import rearrange, repeat

from wm_basicsr.archs.vqgan_arch import Encoder, VectorQuantizer, GumbelQuantizer, Generator, ResBlock
from wm_basicsr.archs.arch_util import flow_warp, resize_flow
from wm_basicsr.archs.gmflow_arch import FlowGenerator
from wm_basicsr.utils import get_root_logger
from wm_basicsr.utils.registry import ARCH_REGISTRY

from diffusers.models.attention import FeedForward, AdaLayerNorm

# gpu_tracker = MemTracker()

class CrossAttention(nn.Module):
    r"""
    copy from diffuser 0.11.1
    A cross attention layer.
    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        use_relative_position: bool = False,
    ):
        super().__init__()
        # print('num head', heads)
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.scale = dim_head**-0.5

        self.heads = heads
        self.dim_head = dim_head
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads
        self._slice_size = None
        self._use_memory_efficient_attention_xformers = False
        self.added_kv_proj_dim = added_kv_proj_dim

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.group_norm = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

        # print(use_relative_position)
        self.use_relative_position = use_relative_position
        if self.use_relative_position:
            self.rotary_emb = RotaryEmbedding(min(32, dim_head))
        #     # print(dim_head)
        #     # print(heads)
        #     # adopt https://github.com/huggingface/transformers/blob/8a817e1ecac6a420b1bdc701fcc33535a3b96ff5/src/transformers/models/bert/modeling_bert.py#L265
        #     self.max_position_embeddings = 32
        #     self.distance_embedding = nn.Embedding(2 * self.max_position_embeddings - 1, dim_head)

        #     self.dropout = nn.Dropout(dropout)


    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor
    
    def reshape_for_scores(self, tensor):
        # split heads and dims
        # tensor should be [b (h w)] f (d nd)
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        return tensor
    
    def same_batch_dim_to_heads(self, tensor):
        batch_size, head_size, seq_len, dim = tensor.shape # [b (h w)] nd f d
        tensor = tensor.reshape(batch_size, seq_len, dim * head_size)
        return tensor

    def set_attention_slice(self, slice_size):
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        self._slice_size = slice_size

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, use_image_num=None):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states) # [b (h w)] f (nd * d)

        # print('before reshpape query shape', query.shape)
        dim = query.shape[-1]
        if not self.use_relative_position:
            query = self.reshape_heads_to_batch_dim(query) # [b (h w) nd] f d
        # print('after reshape query shape', query.shape)

        if self.added_kv_proj_dim is not None:
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)
            encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)
            encoder_hidden_states_key_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
            encoder_hidden_states_value_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

            key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)
            
            if not self.use_relative_position:
                key = self.reshape_heads_to_batch_dim(key)
                value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


    def _attention(self, query, key, value, attention_mask=None):
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )

        # print('query shape', query.shape)
        # print('key shape', key.shape)
        # print('value shape', value.shape)

        if attention_mask is not None:
            # print('attention_mask', attention_mask.shape)
            # print('attention_scores', attention_scores.shape)
            # exit()
            attention_scores = attention_scores + attention_mask

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        # print(attention_probs.shape)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)
        # print(attention_probs.shape)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)
        # print(hidden_states.shape)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        # print(hidden_states.shape)
        # exit()
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim, attention_mask):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]

            if self.upcast_attention:
                query_slice = query_slice.float()
                key_slice = key_slice.float()

            attn_slice = torch.baddbmm(
                torch.empty(slice_size, query.shape[1], key.shape[1], dtype=query_slice.dtype, device=query.device),
                query_slice,
                key_slice.transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )

            if attention_mask is not None:
                attn_slice = attn_slice + attention_mask[start_idx:end_idx]

            if self.upcast_softmax:
                attn_slice = attn_slice.float()

            attn_slice = attn_slice.softmax(dim=-1)

            # cast back to the original dtype
            attn_slice = attn_slice.to(value.dtype)
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.

    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.

    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.

    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)
                       ) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)),
                               device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            nn.init.xavier_uniform_(module.out_proj.weight)
            if module.in_proj_bias is not None:
                nn.init.constant_(module.in_proj_bias, 0.)
                nn.init.constant_(module.out_proj.bias, 0.)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        # self attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2*in_ch, out_ch)

        self.scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, enc_feat, dec_feat, w=1):
        # print(enc_feat.shape, dec_feat.shape)
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + residual
        return out


class CrossFrameFusionLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.use_ada_layer_norm = num_embeds_ada_norm is not None

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout,
                              activation_fn=activation_fn)

        # Cross Frame Attention
        self.attn = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        nn.init.zeros_(self.attn.to_out[0].weight.data)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(self, curr_states, prev_states, residual=True):
        B, C, H, W = curr_states.shape
        curr_states = rearrange(curr_states, "b c h w -> b (h w) c")
        prev_states = rearrange(prev_states, "b c h w -> b (h w) c")

        if residual:
            res = curr_states

        curr_states = self.attn(curr_states, prev_states)
        curr_states = self.norm1(curr_states)

        if residual:
            curr_states = curr_states + res
            res = curr_states

        curr_states = self.ff(curr_states)
        curr_states = self.norm2(curr_states)

        if residual:
            curr_states = curr_states + res

        curr_states = rearrange(curr_states, "b (h w) c -> b c h w", h=H)
        return curr_states


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None

        # SC-Attn
        self.attn1 = SparseCausalAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )
        self.norm1 = AdaLayerNorm(
            dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # # Cross-Attn
        # if cross_attention_dim is not None:
        #     self.attn2 = CrossAttention(
        #         query_dim=dim,
        #         cross_attention_dim=cross_attention_dim,
        #         heads=num_attention_heads,
        #         dim_head=attention_head_dim,
        #         dropout=dropout,
        #         bias=attention_bias,
        #         upcast_attention=upcast_attention,
        #     )
        # else:
        #     self.attn2 = None

        # if cross_attention_dim is not None:
        #     self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        # else:
        #     self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout,
                              activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

        # Temp-Attn
        self.attn_temp = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
        self.norm_temp = AdaLayerNorm(
            dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if self.attn2 is not None:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            # self.attn_temp._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None):
        # SparseCausal-Attention
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(
                hidden_states)
        )

        if self.only_cross_attention:
            hidden_states = (
                self.attn1(norm_hidden_states, encoder_hidden_states,
                           attention_mask=attention_mask) + hidden_states
            )
        else:
            hidden_states = self.attn1(
                norm_hidden_states, attention_mask=attention_mask, video_length=video_length) + hidden_states

        # if self.attn2 is not None:
        #     # Cross-Attention
        #     norm_hidden_states = (
        #         self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
        #     )
        #     hidden_states = (
        #         self.attn2(
        #             norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        #         )
        #         + hidden_states
        #     )

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # Temporal-Attention
        d = hidden_states.shape[1]
        hidden_states = rearrange(
            hidden_states, "(b f) d c -> (b d) f c", f=video_length)
        norm_hidden_states = (
            self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(
                hidden_states)
        )
        hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states


class SparseCausalAttention(CrossAttention):
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        batch_size, sequence_length, _ = hidden_states.shape

        if self.group_norm is not None:
            hidden_states = self.group_norm(
                hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        former_frame_index = torch.arange(video_length) - 1
        former_frame_index[0] = 0

        # d = h*w
        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        key = torch.cat([key[:, [0] * video_length],
                        key[:, former_frame_index]], dim=2)
        key = rearrange(key, "b f d c -> (b f) d c")

        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        value = torch.cat([value[:, [0] * video_length],
                          value[:, former_frame_index]], dim=2)
        value = rearrange(value, "b f d c -> (b f) d c")

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(
                    attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(
                    self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(
                query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(
                    query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(
                    query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class KalmanFilter(nn.Module):
    def __init__(self, emb_dim, num_attention_heads,
                 attention_head_dim, num_uncertainty_layers):
        super().__init__()
        self.uncertainty_estimator = nn.ModuleList(
            [
                BasicTransformerBlock(
                    emb_dim,
                    num_attention_heads,
                    attention_head_dim,
                )
                for d in range(num_uncertainty_layers)
            ]
        )

        self.kalman_gain_calculator = nn.Sequential(
            ResBlock(emb_dim, emb_dim),
            ResBlock(emb_dim, emb_dim),
            ResBlock(emb_dim, emb_dim),
            nn.Conv2d(emb_dim, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def predict(self, z_hat, flow):
        # Predict the next state based on the current state and flow (if available)
        flow = rearrange(flow, "n c h w -> n h w c")
        z_prime = flow_warp(z_hat, flow)
        return z_prime

    def update(self, z_code, z_prime, gain):
        # Update the state and uncertainty based on the measurement and Kalman gain
        z_hat = (1 - gain) * z_code + gain * z_prime
        return z_hat

    def calc_gain(self, z_codes):
        assert z_codes.dim(
        ) == 5, f"Expected z_codes to have ndim=5, but got ndim={z_codes.dim()}."
        video_length = z_codes.shape[1]
        height, width = z_codes.shape[3:5]

        # Assume input shape of uncertainty_estimator to be [(b f) d c]
        z_tmp = rearrange(z_codes, "b f c h w -> (b f) (h w) c")
        h_codes = z_tmp
        for block in self.uncertainty_estimator:
            h_codes = block(h_codes, video_length=video_length)

        h_codes = rearrange(
            h_codes, "(b f) (h w) c -> (b f) c h w", h=height, f=video_length)
        w_codes = self.kalman_gain_calculator(h_codes)

        w_codes = rearrange(
            w_codes, "(b f) c h w -> b f c h w", f=video_length)

        # pdb.set_trace()
        return w_codes


def load_vqgan_checkpoint(model, vqgan_path, logger=None):
    """Load VQGAN checkpoint into model components.
    
    Args:
        model: The model to load weights into
        vqgan_path (str): Path to the VQGAN checkpoint
        logger: Logger instance
    """
    if logger is None:
        logger = get_root_logger()
        
    # Load VQGAN checkpoint, load params_ema or params
    ckpt = torch.load(vqgan_path, map_location='cpu', weights_only=True)
    if 'params_ema' in ckpt:
        state_dict = ckpt['params_ema']
        logger.info(f'Loading VQGAN from: {vqgan_path} [params_ema]')
    elif 'params' in ckpt:
        state_dict = ckpt['params']
        logger.info(f'Loading VQGAN from: {vqgan_path} [params]')
    else:
        raise ValueError(f'Wrong params in checkpoint: {vqgan_path}')
    
    # Load encoder weights into both encoders
    encoder_state_dict = {k.split('encoder.')[-1]: v for k, v in state_dict.items() if k.startswith('encoder.')}
    model.encoder.load_state_dict(encoder_state_dict, strict=True)
    model.hq_encoder.load_state_dict(encoder_state_dict, strict=True)
    
    # Load quantizer weights
    quantizer_state_dict = {k.split('quantize.')[-1]: v for k, v in state_dict.items() if k.startswith('quantize.')}
    model.quantize.load_state_dict(quantizer_state_dict, strict=True)
    
    # Load generator weights 
    generator_state_dict = {k.split('generator.')[-1]: v for k, v in state_dict.items() if k.startswith('generator.')}
    model.generator.load_state_dict(generator_state_dict, strict=True)


@ARCH_REGISTRY.register()
class KEEP(nn.Module):
    def __init__(self, img_size=512, nf=64, ch_mult=[1, 2, 2, 4, 4, 8], quantizer_type="nearest",
                 res_blocks=2, attn_resolutions=[16], codebook_size=1024, emb_dim=256,
                 beta=0.25, gumbel_straight_through=False, gumbel_kl_weight=1e-8, vqgan_path=None,
                 dim_embd=512, n_head=8, n_layers=9, latent_size=256,
                 cft_list=['32', '64', '128', '256'], fix_modules=['quantize', 'generator'],
                 flownet_path=None, kalman_attn_head_dim=64, num_uncertainty_layers=4,
                 cond=1, cfa_list=[], cfa_nhead=4, cfa_dim=256,
                 cfa_nlayers=4, cross_residual=True,
                 temp_reg_list=[], mask_ratio=0.):
        super().__init__()

        self.cond = cond
        self.cft_list = cft_list
        self.cfa_list = cfa_list
        self.temp_reg_list = temp_reg_list
        self.use_residual = cross_residual
        self.mask_ratio = mask_ratio
        self.latent_size = latent_size
        logger = get_root_logger()

        # alignment
        self.flownet = FlowGenerator(path=flownet_path)

        # Kalman Filter
        self.kalman_filter = KalmanFilter(
            emb_dim=emb_dim,
            num_attention_heads=n_head,
            attention_head_dim=kalman_attn_head_dim,
            num_uncertainty_layers=num_uncertainty_layers,
        )

        # Create encoders with same architecture
        encoder_config = dict(
            in_channels=3,
            nf=nf,
            emb_dim=emb_dim,
            ch_mult=ch_mult,
            num_res_blocks=res_blocks,
            resolution=img_size,
            attn_resolutions=attn_resolutions
        )
        
        self.hq_encoder = Encoder(**encoder_config)
        self.encoder = Encoder(**encoder_config)

        # VQGAN components
        if quantizer_type == "nearest":
            self.quantize = VectorQuantizer(codebook_size, emb_dim, beta)
        elif quantizer_type == "gumbel":
            self.quantize = GumbelQuantizer(
                codebook_size, emb_dim, emb_dim, gumbel_straight_through, gumbel_kl_weight
            )
            
        self.generator = Generator(
            nf=nf,
            emb_dim=emb_dim,
            ch_mult=ch_mult,
            res_blocks=res_blocks,
            img_size=img_size,
            attn_resolutions=attn_resolutions
        )

        # Load VQGAN checkpoint if provided
        if vqgan_path is not None:
            load_vqgan_checkpoint(self, vqgan_path, logger)

        self.position_emb = nn.Parameter(torch.zeros(latent_size, dim_embd))
        self.feat_emb = nn.Linear(emb_dim, dim_embd)

        # transformer
        self.ft_layers = nn.Sequential(*[TransformerSALayer(embed_dim=dim_embd, nhead=n_head,
                                                            dim_mlp=dim_embd*2, dropout=0.0) for _ in range(n_layers)])

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False))

        self.channels = {
            '16': 512,
            '32': 256,
            '64': 256,
            '128': 128,
            '256': 128,
            '512': 64,
        }

        # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {
            '512': 2, '256': 5, '128': 8, '64': 11, '32': 14, '16': 18}
        # after first residual block for > 16, before attn layer for ==16
        self.fuse_generator_block = {
            '16': 6, '32': 9, '64': 12, '128': 15, '256': 18, '512': 21}

        # cross frame attention fusion
        self.cfa = nn.ModuleDict()
        for f_size in self.cfa_list:
            in_ch = self.channels[f_size]
            self.cfa[f_size] = CrossFrameFusionLayer(dim=in_ch,
                                                     num_attention_heads=cfa_nhead,
                                                     attention_head_dim=cfa_dim)

        # Controllable Feature Transformation (CFT)
        self.cft = nn.ModuleDict()
        for f_size in self.cft_list:
            in_ch = self.channels[f_size]
            self.cft[f_size] = Fuse_sft_block(in_ch, in_ch)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False


    def get_flow(self, x):
        b, t, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        # Forward flow
        with torch.no_grad():
            flows = self.flownet(x_2, x_1).view(b, t - 1, 2, h, w)

        return flows.detach()

    def mask_by_ratio(self, x, mask_ratio=0.):
        if mask_ratio == 0:
            return x

        # B F C H W
        b, t, c, h, w = x.size()
        d = h * w
        x = rearrange(x, "b f c h w -> b f (h w) c")

        len_keep = int(d * (1 - mask_ratio))
        sample = torch.rand((b, t, d, 1), device=x.device).topk(
            len_keep, dim=2).indices
        mask = torch.zeros((b, t, d, 1), dtype=torch.bool, device=x.device)
        mask.scatter_(dim=2, index=sample, value=True)

        x = mask * x
        x = rearrange(x, "b f (h w) c -> b f c h w", h=h)

        return x

    def forward(self, x, detach_16=True, early_feat=True, need_upscale=True):
        """Forward function for KEEP.

        Args:
            lqs (tensor): Input low quality (LQ) sequence of
                shape (b, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (b, t, c, 4h, 4w).
        """
        video_length = x.shape[1]

        if need_upscale:
            x = rearrange(x, "b f c h w -> (b f) c h w")
            x = F.interpolate(x, scale_factor=4, mode='bilinear')
            x = rearrange(x, "(b f) c h w -> b f c h w", f=video_length)

        b, t, c, h, w = x.size()
        flows = self.get_flow(x)  # (B, t-1, 2, H , W)

        # ################### Encoder #####################
        # BTCHW -> (BT)CHW
        x = x.reshape(-1, c, h, w)
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size]
                    for f_size in self.cft_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = rearrange(x, "(b f) c h w -> b f c h w", f=t).detach()

        lq_feat = x

        # gpu_tracker.track('After encoder')
        # ################### Kalman Filter ###############
        z_codes = rearrange(x, "(b f) c h w -> b f c h w", f=t)
        if self.training:
            z_codes = self.mask_by_ratio(z_codes, self.mask_ratio)
        gains = self.kalman_filter.calc_gain(z_codes)

        outs = []
        logits = []
        cross_prev_feat = {}
        gen_feat_dict = defaultdict(list)

        cft_list = [self.fuse_generator_block[f_size]
                     for f_size in self.cft_list]

        cfa_list = [self.fuse_generator_block[f_size]
                    for f_size in self.cfa_list]

        temp_reg_list = [self.fuse_generator_block[f_size]
                    for f_size in self.temp_reg_list]

        for i in range(video_length):
            # print(f'Frame {i} ...')
            if i == 0:
                z_hat = z_codes[:, i, ...]
            else:
                z_prime = self.hq_encoder(
                    self.kalman_filter.predict(prev_out.detach(), flows[:, i-1, ...]))
                z_hat = self.kalman_filter.update(
                    z_codes[:, i, ...], z_prime, gains[:, i, ...])

            # ################# Transformer ###################
            pos_emb = self.position_emb.unsqueeze(1).repeat(1, b, 1)
            # BCHW -> BC(HW) -> (HW)BC
            query_emb = self.feat_emb(z_hat.flatten(2).permute(2, 0, 1))
            for layer in self.ft_layers:
                query_emb = layer(query_emb, query_pos=pos_emb)

            # output logits
            logit = self.idx_pred_layer(query_emb).permute(
                1, 0, 2)  # (hw)bn -> b(hw)n
            logits.append(logit)

            # ################# Quantization ###################
            code_h = int(np.sqrt(self.latent_size))
            soft_one_hot = F.softmax(logit, dim=2)
            _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
            quant_feat = self.quantize.get_codebook_feat(
                top_idx, shape=[b, code_h, code_h, 256])

            if detach_16:
                # for training stage III
                quant_feat = quant_feat.detach()
            else:
                # preserve gradients for stage II
                quant_feat = query_emb + (quant_feat - query_emb).detach()

            # ################## Generator ####################
            x = quant_feat

            for j, block in enumerate(self.generator.blocks):
                x = block(x)

                if j in cft_list:  # fuse after i-th block
                    f_size = str(x.shape[-1])
                    # pdb.set_trace()
                    x = self.cft[f_size](
                        enc_feat_dict[f_size][:, i, ...], x, self.cond)

                if j in cfa_list:
                    f_size = str(x.shape[-1])

                    if i == 0:
                        cross_prev_feat[f_size] = x
                        # print(f_size)
                    else:
                        # pdb.set_trace()
                        prev_fea = cross_prev_feat[f_size]
                        x = self.cfa[f_size](
                            x, prev_fea, residual=self.use_residual)
                        cross_prev_feat[f_size] = x

                if j in temp_reg_list:
                    f_size = str(x.shape[-1])
                    gen_feat_dict[f_size].append(x)

            prev_out = x  # B C H W
            outs.append(prev_out)

        for f_size, feat in gen_feat_dict.items():
            gen_feat_dict[f_size] = torch.stack(feat, dim=1)  # bfchw

        # Convert defaultdict to regular dict before returning
        gen_feat_dict = dict(gen_feat_dict)

        logits = torch.stack(logits, dim=1)  # b(hw)n -> bf(hw)n
        logits = rearrange(logits, "b f l n -> (b f) l n")
        outs = torch.stack(outs, dim=1)  # bfchw
        if self.training:
            if early_feat:
                return outs, logits, lq_feat, gen_feat_dict
            else:
                return outs, gen_feat_dict
        else:
            return outs


def count_parameters(model):
    # Initialize counters
    total_params = 0
    sub_module_params = {}

    # Loop through all the modules in the model
    for name, module in model.named_children():
        # if len(list(module.children())) == 0:  # Check if it's a leaf module
        params = sum(p.numel() for p in module.parameters())
        total_params += params
        sub_module_params[name] = params

    return total_params, sub_module_params


if __name__ == '__main__':
    import time
    batch_size = 1
    video_length = 4
    height = 128
    width = 128

    model = KEEP(
        img_size=512,
        emb_dim=256,
        ch_mult=[1, 2, 2, 4, 4, 8],
        dim_embd=512,
        n_head=8,
        n_layers=4,
        codebook_size=1024,
        cft_list=[],
        fix_modules=['generator', 'quantize', 'flownet', 'cft', 'hq_encoder',
                     'encoder', 'feat_emb', 'ft_layers', 'idx_pred_layer'],
        flownet_path="../../weights/GMFlow/gmflow_sintel-0c07dcb3.pth",
        kalman_attn_head_dim=32,
        num_uncertainty_layers=3,
        cond=0,
        cfa_list=['32'],
        cfa_nhead=4,
        cfa_dim=256,
        temp_reg_list=['64'],
    ).cuda()

    total_params = sum(map(lambda x: x.numel(), model.parameters()))
    print(f"Total parameters in the model: {total_params / 1e6:.2f} M")

    dummy_input = torch.randn((1, 20, 3, 128, 128)).cuda()

    start_time = time.time()

    with torch.no_grad():
        for _ in range(100):
            out = model(dummy_input)
    elapsed_time = time.time() - start_time

    print(f"Forward pass time: {elapsed_time / 100 / 20 * 1000:.2f} ms")
    print(out.shape)
