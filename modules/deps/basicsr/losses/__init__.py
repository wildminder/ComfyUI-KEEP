# https://github.com/xinntao/BasicSR
# flake8: noqa
from copy import deepcopy

from basicsr.utils import get_root_logger
from basicsr.utils.registry import LOSS_REGISTRY
from .gan_loss import g_path_regularize, gradient_penalty_loss, r1_penalty

# P2.4 - explicit import list; the eager folder scan is gone
from . import (
    basic_loss,
    gan_loss,
)

__all__ = ['build_loss', 'gradient_penalty_loss', 'r1_penalty', 'g_path_regularize']


def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Loss type.
    """
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
