"""KEEP image sequence node (V3 schema - P4.3).

The node id (``KEEP_ProcessImageSequence``), the display name ("KEEP Image
Sequence"), the category (``ComfyUI-KEEP``), the socket types and every
widget (name, default, min/max/step and order - including
``max_clip_length``) are compatibility guarantees of the pack - the V3 schema
below carries exactly the contract of the former V1 ``INPUT_TYPES``.

The sequence path is the only path that needs the **optical flow** of KEEP
(the propagation of the features along the time axis).  The node therefore
calls ``pack.ensure_flow()`` before the processing of a multi frame run -
the hook is idempotent, and P5.5 implements the resolution of the GMFlow
checkpoint behind it.
"""

from comfy_api.latest import io

try:  # the package import (ComfyUI loads the node package)
    from ..src.keep_model_loader import KEEPModelPack
    from ..src.keep_processor import KEEPFaceProcessor
except ImportError:  # the flat import (the repository root on sys.path)
    from src.keep_model_loader import KEEPModelPack
    from src.keep_processor import KEEPFaceProcessor


class KEEP_ProcessImageSequenceNode(io.ComfyNode):
    """Restore the faces of a sequence of frames with KEEP."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KEEP_ProcessImageSequence",
            display_name="KEEP Image Sequence",
            category="ComfyUI-KEEP",
            description=(
                "Restore and enhance the faces of a sequence of frames (e.g. "
                "the frames of a video) with the KEEP face restoration model."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The frames whose faces are restored.",
                ),
                io.Custom("KEEP_MODEL_PACK").Input(
                    "keep_model",
                    tooltip="The model pack of the 'Load KEEP Models' node.",
                ),
                io.Float.Input(
                    "final_upscale_factor",
                    default=1.0,
                    min=0.5,
                    max=4.0,
                    step=0.1,
                    tooltip=(
                        "The final upscaling factor for the output frames. They "
                        "will be resized to this scale after processing."
                    ),
                ),
                io.Boolean.Input(
                    "has_aligned_frames",
                    default=False,
                    tooltip="Check if the input frames are already aligned 512x512 faces.",
                ),
                io.Boolean.Input(
                    "only_center_face",
                    default=True,
                    tooltip=(
                        "If frames have multiple faces, only process the one "
                        "closest to the center."
                    ),
                ),
                io.Boolean.Input(
                    "draw_bounding_box",
                    default=False,
                    tooltip=(
                        "Draw a bounding box around the detected face on the "
                        "output frames."
                    ),
                ),
                io.Int.Input(
                    "max_clip_length",
                    default=20,
                    min=1,
                    max=100,
                    step=1,
                    tooltip=(
                        "Maximum number of frames to process in a single batch "
                        "to manage VRAM."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    id="processed_images",
                    display_name="processed_images",
                    tooltip="The restored frames.",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        """Refuse a model pack that is not a ``KEEPModelPack``."""
        keep_model = kwargs.get("keep_model")
        if not isinstance(keep_model, KEEPModelPack):
            return (
                "Invalid KEEP Model Pack provided: expected a KEEPModelPack, "
                f"got {type(keep_model).__name__}. Connect the output of the "
                "'Load KEEP Models' node."
            )
        return True

    @classmethod
    def execute(
        cls,
        images,
        keep_model,
        final_upscale_factor,
        has_aligned_frames,
        only_center_face,
        draw_bounding_box,
        max_clip_length,
    ) -> io.NodeOutput:
        """Restore the frames of the sequence.

        The pack is always offloaded - also when the processing fails (the
        exception is **not** swallowed: a failed node must fail loudly).
        """
        try:
            num_frames = int(images.shape[0]) if hasattr(images, "shape") else len(images)

            keep_model.load_device()
            if num_frames > 1:
                # the optical flow is only used by a multi frame run - the
                # hook is idempotent and a no-op until P5.5 resolves the
                # GMFlow checkpoint
                keep_model.ensure_flow()

            processor = KEEPFaceProcessor(keep_model)
            processed_images_tensor = processor.process_image_sequence(
                image_sequence_tensor=images,
                final_upscale_factor=final_upscale_factor,
                has_aligned_frames=has_aligned_frames,
                only_center_face=only_center_face,
                draw_box=draw_bounding_box,
                max_clip_length=max_clip_length,
            )
            return io.NodeOutput(processed_images_tensor)
        finally:
            keep_model.offload()
