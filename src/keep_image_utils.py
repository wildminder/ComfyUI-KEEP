"""The conversions between the ComfyUI IMAGE tensors and the cv2 images.

The functions moved here from ``modules/utils.py`` in P3.2 - they are the
image half of the old utility module, the config half lives in
``src/keep_model_configs.py``.

The dependency slimming replaced the channel conversions of cv2 with torch
indexing and added the torch bilinear ``resize_bgr_u8`` for the INTER_LINEAR
resizes of the processor; the equivalences against the cv2 ops are pinned in
``tests/test_image_op_equivalence.py``.

The GPU video pipeline redesign added the **batch** forms of the conversions
(``*_batch``): the same elementwise math as the per-frame functions, run on
the whole stack with a single host transfer instead of one per frame
(design sections 7.1 / 7.4 / 7.8 of
``.dev/docs/2026-10-25-keep-gpu-video-pipeline.md``); their equivalences are
pinned in ``tests/test_gpu_batch_equivalence.py``.

Round 17 (the final stage RAM) added the **lazy** BGR frame source
(:class:`BgrFrameSource`): the paste stage converts the background rows on
demand instead of building a whole-clip uint8 stack.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def comfy_image_to_cv2(comfy_image: torch.Tensor) -> np.ndarray:
    """A ComfyUI IMAGE tensor ``(B, H, W, C)`` -> a cv2 BGR image."""
    if comfy_image.ndim == 3:
        comfy_image = comfy_image.unsqueeze(0)
    # The scale, the uint8 cast and the RGB -> BGR flip run on the tensor at
    # its device (the cast truncates like the numpy ``astype`` it replaced) -
    # the copy to the host happens once, as uint8.
    img_t = (comfy_image.squeeze(0) * 255).to(torch.uint8)
    return img_t[..., [2, 1, 0]].cpu().numpy()


def cv2_to_comfy_image(cv2_image: np.ndarray) -> torch.Tensor:
    """A cv2 BGR image -> a ComfyUI IMAGE tensor ``(1, H, W, C)``."""
    bgr_t = torch.from_numpy(np.ascontiguousarray(cv2_image))
    rgb_t = bgr_t[..., [2, 1, 0]].to(torch.float32) / 255.0
    return rgb_t.unsqueeze(0)


def resize_bgr_u8(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a uint8 BGR image ``(H, W, 3)`` with torch bilinear filtering.

    The torch-native replacement of the INTER_LINEAR resizes of the
    processor: ``size`` keeps the cv2 ``(width, height)`` order, both
    libraries share the half-pixel convention (``align_corners=False``) and
    the measured residual against cv2 is at most 1/255 (the uint8 rounding)
    over the geometry matrix of ``tests/test_image_op_equivalence.py``, with
    an identity resize being exact.
    """
    height, width = size[1], size[0]
    tensor = torch.from_numpy(np.ascontiguousarray(image))
    resized = F.interpolate(
        tensor.permute(2, 0, 1).unsqueeze(0).float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    return (
        resized.squeeze(0)
        .permute(1, 2, 0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .contiguous()
        .numpy()
    )


# ---------------------------------------------------------------------------
# The batch forms of the conversions (the GPU video pipeline redesign)
# ---------------------------------------------------------------------------

#: The pixel budget of one batch-conversion chunk: the ``* 255`` float32
#: transient is four bytes per value (twelve per RGB pixel), so a chunk
#: holds at most this many pixels of it.  The same pixel-bound convention
#: as ``src/keep_gpu_paste.py``'s ``GRAY_CHUNK_PIXELS`` (``4 << 20``) so one
#: canvas size means one bound everywhere.
COMFY_BATCH_CHUNK_PIXELS = 4 << 20


def comfy_batch_to_bgr_u8(comfy_batch: torch.Tensor) -> np.ndarray:
    """A ComfyUI IMAGE batch ``(N, H, W, C)`` -> ``(N, H, W, 3)`` uint8 BGR.

    The batch form of :func:`comfy_image_to_cv2` (design section 7.1): the
    scale, the uint8 cast and the RGB -> BGR flip run on the tensor at its
    device and the copy to the host happens **once** for the whole batch,
    where the per-frame loop of the sequence path made one transfer per
    frame.  The ops are the same elementwise math, so the output is bit-exact
    against the per-frame loop (pinned in ``tests/test_gpu_batch_equivalence.py``).

    The ``* 255`` float32 transient is chunked over the frame axis against
    :data:`COMFY_BATCH_CHUNK_PIXELS` so no whole-batch float32 exists; the
    elementwise chain is identical per group, so the bytes are unchanged.
    """
    if comfy_batch.ndim != 4:
        raise ValueError(
            f"comfy_batch_to_bgr_u8 expects (N, H, W, C), got {tuple(comfy_batch.shape)}"
        )
    num, height, width = comfy_batch.shape[:3]
    num, height, width = int(num), int(height), int(width)
    pixels = max(1, height * width)
    chunk = max(1, COMFY_BATCH_CHUNK_PIXELS // pixels)
    out = torch.empty((num, height, width, 3), dtype=torch.uint8, device=comfy_batch.device)
    for start in range(0, num, chunk):
        stop = min(start + chunk, num)
        block = (comfy_batch[start:stop] * 255).to(torch.uint8)
        out[start:stop] = block[..., [2, 1, 0]]
    return out.cpu().numpy()


def bgr_u8_batch_to_comfy(u8_batch: np.ndarray | torch.Tensor) -> torch.Tensor:
    """A uint8 BGR batch ``(N, H, W, 3)`` -> a ComfyUI IMAGE batch ``(N, H, W, 3)`` float32.

    The batch form of :func:`cv2_to_comfy_image` (design section 7.8): a
    numpy array or a torch uint8 tensor on any device is accepted; the flip
    and the /255 scale run where the input lives and the result is returned
    on the CPU - one transfer for the whole batch, where the per-frame loop
    made N.  Bit-exact against the per-frame loop.
    """
    if isinstance(u8_batch, np.ndarray):
        u8_batch = torch.from_numpy(np.ascontiguousarray(u8_batch))
    rgb = u8_batch[..., [2, 1, 0]].to(torch.float32) / 255.0
    return rgb.cpu()


def bgr_u8_batch_to_normalized(u8_bchw: torch.Tensor) -> torch.Tensor:
    """A uint8 BGR face stack ``(N, 3, H, W)`` -> the KEEP input ``(N, 3, H, W)`` in [-1, 1].

    The batch form of the crop tail ``img2tensor(cropped / 255., bgr2rgb=True,
    float32=True)`` + ``(t - 0.5) / 0.5``: the BGR -> RGB flip, the /255
    scale and the normalize, batched on the input's device.

    The whole chain runs in **float32** (round 6, Opt 3).  The earlier form
    divided in float64 and cast once to reproduce ``img2tensor``'s numpy
    float64 division bit for bit; the owner's recorded decision (the plan of
    ``.dev/docs/2026-10-29-keep-owner-fix1-plan.md``) is that a **one-ulp
    float32** difference - a single rounding of the ``u8 / 255`` quotient,
    magnified at most by a factor of two by the final ``div_(0.5)`` - does
    not justify the 1:64 memory ratio of the transient float64 stack
    (12 bytes instead of 4 per element of the chunk's crop).  The /255 of a
    uint8 value is either exactly representable or needs one rounding; the
    float64 pipeline is only ever *different*, never more correct.

    The fp32 value is pinned twice in ``tests/test_gpu_batch_equivalence.py``:
    against the vendored ``img2tensor`` chain within the documented
    ``2**-22`` tolerance, and against the explicit float32 chain bit for
    bit (the exact operator order and the output dtype).
    """
    rgb = u8_bchw[:, [2, 1, 0]]
    return rgb.to(torch.float32).div_(255.0).sub_(0.5).div_(0.5)


# ---------------------------------------------------------------------------
# The lazy BGR frame source (round 17 - the final stage RAM)
# ---------------------------------------------------------------------------


class BgrFrameSource:
    """A lazy source of the uint8 BGR frames of a ComfyUI IMAGE batch.

    The paste stage needs the clip's background frames **per group**, not
    as one whole-clip uint8 stack - on the owner's 777-frame 1500x1080 clip
    that stack was ~3.78 GB of host RAM (round 17).  This source wraps the
    caller's float32 IMAGE tensor (no copy - the tensor the node already
    holds) and converts the rows **on demand** through
    :func:`comfy_batch_to_bgr_u8` (already chunked over the pixel budget,
    :data:`COMFY_BATCH_CHUNK_PIXELS`).

    The conversion is elementwise - the ``* 255`` scale, the uint8 cast and
    the RGB -> BGR flip - so a row converted alone is **byte-identical** to
    the same row of the whole-clip conversion.  That identity is pinned in
    ``tests/test_gpu_batch_equivalence.py``.

    The source speaks the protocol the paste core consumes: ``.shape`` (the
    shape of the materialized stack), ``len()``, ``.group(indices)`` (the
    asked rows), ``.row(index)`` (one row) and ``.select(indices)`` (a lazy
    view of a subset of the rows - the view converts nothing until one of
    its rows is asked for).  ``.materialize()`` builds the whole stack for
    the modes that need it (the full and host paste modes) - the disclosed
    limitation of the lazy path.

    Args:
        batch: the ComfyUI IMAGE batch ``(N, H, W, 3)`` - the float32
            tensor of the clip (the node's input).
        convert: the optional converter seam - a callable
            ``(torch.Tensor) -> np.ndarray``.  The processor injects its
            own module global there, so a spy on the processor's converter
            sees every conversion of a run; the default resolves
            :func:`comfy_batch_to_bgr_u8` **at call time** (so a patched
            module global is observed too).
    """

    def __init__(
        self,
        batch: torch.Tensor,
        convert: Callable[[torch.Tensor], np.ndarray] | None = None,
    ) -> None:
        if batch.ndim != 4:
            raise ValueError(
                "BgrFrameSource expects an IMAGE batch (N, H, W, C), got "
                f"{tuple(batch.shape)}"
            )
        self._batch = batch
        self._convert = convert
        self._rows = np.arange(int(batch.shape[0]), dtype=np.int64)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """The shape of the materialized stack: ``(N, H, W, 3)``."""
        _, height, width = (int(value) for value in self._batch.shape[:3])
        return (len(self), height, width, 3)

    def __len__(self) -> int:
        """The number of rows - the host bench calls ``len()`` on the stack."""
        return int(self._rows.size)

    def group(
        self, indices: Sequence[int] | range | slice | np.ndarray
    ) -> np.ndarray:
        """The asked rows as one uint8 BGR array ``(k, H, W, 3)``.

        The rows are converted **now**, through the converter of the seam -
        the elementwise chain of :func:`comfy_batch_to_bgr_u8`, so the bytes
        are the ones of the whole-clip conversion (pinned in
        ``tests/test_gpu_batch_equivalence.py``).
        """
        converter = (
            self._convert if self._convert is not None else comfy_batch_to_bgr_u8
        )
        return converter(self._gather(self._local_rows(indices)))

    def row(self, index: int) -> np.ndarray:
        """One row as a uint8 BGR image ``(H, W, 3)``."""
        return self.group((index,))[0]

    def select(
        self, indices: Sequence[int] | range | slice | np.ndarray
    ) -> "BgrFrameSource":
        """A lazy view of the selected rows - no conversion, no copy.

        The view maps its **local** indices to the rows of the clip: the
        view's row ``i`` is the clip's row ``indices[i]``.  The paste twins
        use the view to ask for their own eligible rows, so the rows that
        are not pasted are never converted and no gather copy is built.
        """
        view = BgrFrameSource(self._batch, convert=self._convert)
        view._rows = self._local_rows(indices)
        return view

    def materialize(self) -> np.ndarray:
        """The whole (selected) stack as one uint8 BGR array.

        The disclosed limitation of the lazy path: the modes that need a
        whole-batch background (the full and host paste modes) build it
        here - one whole-batch conversion, as before.
        """
        return self.group(range(len(self)))

    def _local_rows(
        self, indices: Sequence[int] | range | slice | np.ndarray
    ) -> np.ndarray:
        """Map this source's local indices to the rows of the clip."""
        if isinstance(indices, (slice, range)):
            return self._rows[indices]
        selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1:
            raise ValueError(
                "the frame indices must be a 1-D sequence of row numbers "
                f"(got {selected.ndim} dimensions); use .row() for one row"
            )
        return self._rows[selected]

    def _gather(self, rows: np.ndarray) -> torch.Tensor:
        """The clip's rows as one tensor - a view when they are contiguous."""
        if rows.size == 0:
            return self._batch[:0]
        if rows.size == 1:
            start = int(rows[0])
            return self._batch[start : start + 1]
        if bool(np.all(np.diff(rows) == 1)):
            start = int(rows[0])
            return self._batch[start : start + int(rows.size)]
        return self._batch[torch.from_numpy(np.ascontiguousarray(rows))]
