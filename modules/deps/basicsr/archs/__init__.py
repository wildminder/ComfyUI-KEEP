# https://github.com/xinntao/BasicSR
# flake8: noqa
# 2026-11-03 keep-fix5-implement1 Phase 4.1: the explicit import list is
# reduced to the four architectures the pack uses.
from copy import deepcopy
from ..utils import get_root_logger
from ..utils.registry import ARCH_REGISTRY

from . import (
    gmflow_arch,
    keep_arch,
    spectral_norm_arch,
    vqgan_arch,
)

__all__ = ['build_network']


def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net
