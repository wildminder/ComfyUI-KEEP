# https://github.com/xinntao/BasicSR
# flake8: noqa
from copy import deepcopy

from ..utils import get_root_logger
from ..utils.registry import MODEL_REGISTRY

# P2.4 - explicit import list; the eager folder scan is gone
from . import (
    base_model,
    edvr_model,
    esrgan_model,
    hifacegan_model,
    keep_gan_model,
    keep_model,
    realesrgan_model,
    realesrnet_model,
    sr_model,
    srgan_model,
    stylegan2_model,
    swinir_model,
    video_base_model,
    video_gan_model,
    video_recurrent_gan_model,
    video_recurrent_model,
)

__all__ = ['build_model']


def build_model(opt):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    opt = deepcopy(opt)
    model = MODEL_REGISTRY.get(opt['model_type'])(opt)
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model
