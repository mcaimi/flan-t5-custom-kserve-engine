#!/usr/bin/env python

import logging
import platform

import torch.cuda as tc
if platform.system() == "Darwin":
    import torch.backends.mps as tmps
from torch import float16, float32

logger = logging.getLogger(__name__)


def get_accelerator_device():
    accelerator = "cpu"
    dtype = float32

    logger.info("Checking for the availability of a GPU...")
    if tc.is_available():
        device_name = tc.get_device_name()
        device_capabilities = tc.get_device_capability()
        device_available_mem, device_total_mem = [x / 1024**3 for x in tc.mem_get_info()]
        logger.info("CUDA GPU available: %s - %s - %.1f/%.1f GB VRAM",
                     device_name, device_capabilities, device_available_mem, device_total_mem)
        accelerator = "cuda"
        dtype = float16
    elif platform.system() == "Darwin" and tmps.is_available():
        device_name = tmps.get_name()
        device_cores = tmps.get_core_count()
        logger.info("Apple MPS available: %s - %d Cores", device_name, device_cores)
        accelerator = "mps"
        dtype = float16

    return accelerator, dtype
