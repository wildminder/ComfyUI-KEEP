# https://github.com/xinntao/BasicSR
# flake8: noqa
from copy import deepcopy

from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY

# P2.4 - explicit import list; the eager folder scan is gone
from . import (
    arcface_arch,
    basicvsr_arch,
    basicvsrpp_arch,
    dfdnet_arch,
    discriminator_arch,
    duf_arch,
    ecbsr_arch,
    edsr_arch,
    edvr_arch,
    gmflow_arch,
    hifacegan_arch,
    keep_arch,
    rcan_arch,
    ridnet_arch,
    rrdbnet_arch,
    spectral_norm_arch,
    spynet_arch,
    srresnet_arch,
    srvgg_arch,
    stylegan2_arch,
    stylegan2_bilinear_arch,
    swinir_arch,
    tof_arch,
    vgg_arch,
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
