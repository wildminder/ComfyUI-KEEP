"""KEEP single image node (V3 schema - P4.2).

The node id (``KEEP_FaceUpscaleImage``), the display name ("KEEP Single
Image"), the category (``ComfyUI-KEEP``), the socket types and every widget
(name, default, min/max/step and order) are compatibility guarantees of the
pack - the V3 schema below carries exactly the contract of the former V1
``INPUT_TYPES``.

Two behaviour changes come with the V3 migration (they fix the breakages
B8 of the plan):

* a foreign model pack is refused by ``validate_inputs`` with a **message**
  (the V1 node logged the problem and returned ``(None,)``),
* ``execute`` **raises** when the processing fails - the node never returns
  a ``None`` image behind an ``IMAGE`` socket.
"""

from comfy_api.latest import io

try:  # the package import (ComfyUI loads the node package)
    from ..src.keep_image_utils import comfy_image_to_cv2, cv2_to_comfy_image
    from ..src.keep_model_loader import KEEPModelPack, keep_model_pack_error
    from ..src.keep_processor import KEEPFaceProcessor
    from ..src.keep_progress import report_progress
except ImportError:  # the flat import (the repository root on sys.path)
    from src.keep_image_utils import comfy_image_to_cv2, cv2_to_comfy_image
    from src.keep_model_loader import KEEPModelPack, keep_model_pack_error
    from src.keep_processor import KEEPFaceProcessor
    from src.keep_progress import report_progress


class KEEP_FaceUpscaleImageNode(io.ComfyNode):
    """Restore the faces of a single image with KEEP."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KEEP_FaceUpscaleImage",
            display_name="KEEP Single Image",
            category="ComfyUI-KEEP",
            description=(
                "Restore and enhance the faces of a single image with the KEEP "
                "face restoration model."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The image whose faces are restored.",
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
                        "The final upscaling factor for the output image. The "
                        "image will be resized to this scale after processing."
                    ),
                ),
                io.Boolean.Input(
                    "has_aligned_face",
                    default=False,
                    tooltip="Check if the input image is an already aligned 512x512 face.",
                ),
                io.Boolean.Input(
                    "only_center_face",
                    default=True,
                    tooltip=(
                        "If the image has multiple faces, only process the one "
                        "closest to the center."
                    ),
                ),
                io.Boolean.Input(
                    "draw_bounding_box",
                    default=False,
                    tooltip=(
                        "Draw a bounding box around the detected face on the "
                        "output image."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    id="IMAGE",
                    tooltip="The restored image.",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        """Refuse a model pack that is not a ``KEEPModelPack``.

        ComfyUI runs ``validate_inputs`` during the **prompt validation**,
        where the links are not resolved yet: a connected pack arrives as
        ``None``.  An unresolved pack is therefore accepted here and checked
        in ``execute``, which raises the actionable error.
        """
        keep_model = kwargs.get("keep_model")
        if keep_model is None or (
            isinstance(keep_model, tuple) and len(keep_model) == 1 and keep_model[0] is None
        ):
            # A linked pack is not resolved yet during the prompt validation:
            # ComfyUI hands it over as ``None`` (after its list unwrapping) or
            # as the ``(None,)`` marker of a missing link.  The real type check
            # happens in ``execute``.
            return True
        if not isinstance(keep_model, KEEPModelPack):
            return keep_model_pack_error(keep_model)
        return True

    @classmethod
    def execute(
        cls,
        image,
        keep_model,
        final_upscale_factor,
        has_aligned_face,
        only_center_face,
        draw_bounding_box,
    ) -> io.NodeOutput:
        """Restore the image.

        The pack is always offloaded - also when the processing fails (the
        exception is **not** swallowed: a failed node must fail loudly).
        """
        error = keep_model_pack_error(keep_model)
        if error is not None:
            raise RuntimeError(error)

        try:
            keep_model.load_device()
            report_progress(0.0, 1.0)

            cv2_img_single = comfy_image_to_cv2(image[0].unsqueeze(0))
            processor = KEEPFaceProcessor(keep_model)
            processed_cv2_img = processor.process_image(
                cv2_image_orig=cv2_img_single,
                final_upscale_factor=final_upscale_factor,
                has_aligned=has_aligned_face,
                only_center_face=only_center_face,
                draw_box=draw_bounding_box,
            )
            output_image_single = cv2_to_comfy_image(processed_cv2_img)
            report_progress(1.0, 1.0)
            return io.NodeOutput(output_image_single)
        finally:
            keep_model.offload()
