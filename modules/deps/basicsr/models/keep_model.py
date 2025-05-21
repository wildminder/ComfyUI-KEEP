from collections import OrderedDict
import torch
import torch.nn.functional as F
import pdb

from einops import rearrange

from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.archs.arch_util import flow_warp, resize_flow

from .video_recurrent_model import VideoRecurrentModel


@MODEL_REGISTRY.register()
class KEEPModel(VideoRecurrentModel):
    """KEEP Model.
    """

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        logger = get_root_logger()

        # # load pretrained VQGAN models
        # load_path = self.opt['path'].get('pretrain_network_vqgan', None)
        # if load_path is not None:
        #     param_key = self.opt['path'].get('param_key_vqgan', 'params')
        #     self.load_network(self.net_g, load_path, False, param_key)

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(
                self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get(
                    'strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses.
        self.hq_feat_loss = train_opt.get('use_hq_feat_loss', False)
        self.feat_loss_weight = train_opt.get('feat_loss_weight', 1.0)
        self.cross_entropy_loss = train_opt.get('cross_entropy_loss', False)
        self.entropy_loss_weight = train_opt.get('entropy_loss_weight', 0.5)

        if self.cross_entropy_loss:
            self.generate_idx_gt = True
            assert self.opt.get(
                'network_vqgan', None) is not None, f'Shoule have network_vqgan config or pre-calculated latent code.'
            self.hq_vqgan_fix = build_network(
                self.opt['network_vqgan']).to(self.device)
            self.hq_vqgan_fix.eval()
            for param in self.hq_vqgan_fix.parameters():
                param.requires_grad = False
            # load_path = self.opt['path'].get('pretrain_network_vqgan', None)
            # assert load_path != None, "Should load pre-trained VQGAN"
            # self.load_network(self.hq_vqgan_fix, load_path, strict=False)
        else:
            self.generate_idx_gt = False
        logger.info(f'Need to generate latent GT code: {self.generate_idx_gt}')

        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.perceptual_type = train_opt['perceptual_opt']['type']
            self.cri_perceptual = build_loss(
                train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('temporal_opt'):
            self.temporal_type = train_opt.get('temporal_warp_type', 'GT')
            self.cri_temporal = build_loss(
                train_opt['temporal_opt']).to(self.device)
        else:
            self.cri_temporal = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        logger = get_root_logger()

        optim_names, freezed_names = [], []
        # optimizer g
        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
                optim_names.append(k)
            else:
                freezed_names.append(k)

        logger.warning(f'--------------- Optimizing Params ---------------.')
        for k in optim_names:
            logger.warning(f'Params {k} will be optimized.')
        logger.warning(f'--------------- Freezing Params ---------------.')
        for k in freezed_names:
            logger.warning(f'Params {k} will be freezed.')


        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(
            optim_type, optim_params_g, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        # optimize net_g
        self.optimizer_g.zero_grad()

        if self.generate_idx_gt:
            with torch.no_grad():
                b, f, c, h, w = self.gt.shape
                x = self.hq_vqgan_fix.encoder(self.gt.reshape(-1, c, h, w))
                _, _, quant_stats = self.hq_vqgan_fix.quantize(x)
                min_encoding_indices = quant_stats['min_encoding_indices']
                self.idx_gt = min_encoding_indices.view(b*f, -1)

        if self.hq_feat_loss or self.cross_entropy_loss:
            self.output, logits, lq_feat, gen_feat_dict = self.net_g(
                self.lq, detach_16=True, early_feat=True)
        else:
            self.output, gen_feat_dict = self.net_g(
                self.lq, detach_16=True, early_feat=False)
            if len(gen_feat_dict) == 0:
                gen_feat_dict['HR'] = self.output

        l_g_total = 0
        loss_dict = OrderedDict()
        # hq_feat_loss
        if self.hq_feat_loss:  # codebook loss
            code_h = lq_feat.shape[-1]
            quant_feat_gt = self.net_g.module.quantize.get_codebook_feat(
                self.idx_gt, shape=[b*f, code_h, code_h, 256])
            l_feat_encoder = torch.mean(
                (quant_feat_gt.detach()-lq_feat)**2) * self.feat_loss_weight
            l_g_total += l_feat_encoder
            loss_dict['l_feat_encoder'] = l_feat_encoder

        # cross_entropy_loss
        if self.cross_entropy_loss:
            # b(hw)n -> bn(hw)
            cross_entropy_loss = F.cross_entropy(logits.permute(
                0, 2, 1), self.idx_gt) * self.entropy_loss_weight
            l_g_total += cross_entropy_loss
            loss_dict['l_cross_entropy'] = cross_entropy_loss

        # Temporal consistency loss
        if self.cri_temporal:
            assert len(
                gen_feat_dict) != 0, "Empty features for temporal regularization."
            with torch.no_grad():
                if self.temporal_type == 'GT':
                    flows = self.net_g.module.get_flow(self.gt).detach()
                    flows = rearrange(flows, "b f c h w -> (b f) c h w")
                elif self.temporal_type == 'HR':
                    flows = self.net_g.module.get_flow(self.output).detach()
                    flows = rearrange(flows, "b f c h w -> (b f) c h w")
                elif self.temporal_type == 'Diff':
                    gt_flows = self.net_g.module.get_flow(self.gt).detach()
                    gt_flows = rearrange(gt_flows, "b f c h w -> (b f) c h w")
                    hr_flows = self.net_g.module.get_flow(self.output).detach()
                    hr_flows = rearrange(hr_flows, "b f c h w -> (b f) c h w")
                else:
                    raise ValueError(
                        f'Unsupported temporal mode: {self.temporal_type}.')

            l_temporal = 0
            for f_size, feat in gen_feat_dict.items():
                b, f, c, h, w = feat.shape

                if self.temporal_type == 'GT' or self.temporal_type == 'HR':
                    flow = resize_flow(flows, 'shape', [h, w])  # B*(T-1) 2 H W
                    flow = rearrange(flow, "b c h w -> b h w c")
                    prev_feat = feat[:, :-1, ...].view(-1, c, h, w)
                    curr_feat = feat[:, 1:, ...].view(-1, c, h, w)
                    warp_feat = flow_warp(prev_feat, flow)
                    l_temporal += self.cri_temporal(curr_feat, warp_feat)
                elif self.temporal_type == 'Diff':
                    gt_flow = resize_flow(gt_flows, 'shape', [
                                          h, w])  # B*(T-1) 2 H W
                    gt_flow = rearrange(gt_flow, "b c h w -> b h w c")
                    hr_flow = resize_flow(hr_flows, 'shape', [
                                          h, w])  # B*(T-1) 2 H W
                    hr_flow = rearrange(hr_flow, "b c h w -> b h w c")

                    prev_feat = feat[:, :-1, ...].view(-1, c, h, w)
                    curr_feat = feat[:, 1:, ...].view(-1, c, h, w)
                    gt_warp_feat = flow_warp(prev_feat, gt_flow)
                    hr_warp_feat = flow_warp(prev_feat, hr_flow)
                    l_temporal += self.cri_temporal(gt_warp_feat, hr_warp_feat)

            l_g_total += l_temporal
            loss_dict['l_temporal'] = l_temporal

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_g_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            B, T, C, H, W = self.gt.shape
            if self.perceptual_type == 'PerceptualLoss':
                l_percep, l_style = self.cri_perceptual(
                    self.output.view(-1, C, H, W), self.gt.view(-1, C, H, W))
                if l_percep is not None:
                    l_g_total += l_percep
                    loss_dict['l_percep'] = l_percep
                if l_style is not None:
                    l_g_total += l_style
                    loss_dict['l_style'] = l_style
            elif self.perceptual_type == 'LPIPSLoss':
                l_percep = self.cri_perceptual(
                    self.output.view(-1, C, H, W), self.gt.view(-1, C, H, W))
                l_g_total += l_percep
                loss_dict['l_percep'] = l_percep

        l_g_total.backward()
        self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)
