#!/usr/bin/env python

import platform

try:
    import torch.cuda as tc
    if platform.system() == "Darwin":
        import torch.backends.mps as tmps
    from torch import float16, float32
except Exception as e:
    print(f"Caught Exception during library loading: {e}")
    raise e

# check for the presence of a gpu
def get_accelerator_device():
    # assume no gpu is present
    accelerator = "cpu"
    dtype = float32

    # test the presence of a GPU...
    print("Checking for the availability of a GPU...")
    if tc.is_available():
        device_name = tc.get_device_name()
        device_capabilities = tc.get_device_capability()
        device_available_mem, device_total_mem = [x / 1024**3 for x in tc.mem_get_info()]
        print(f"A GPU is available! [{device_name} - {device_capabilities} - {device_available_mem}/{device_total_mem} GB VRAM]")
        accelerator = "cuda"
        dtype = float16
    if platform.system() == "Darwin":
        if tmps.is_available():
            device_name = tmps.get_name()
            device_cores = tmps.get_core_count()
            print(f"Apple Device is available! [{device_name} - {device_cores} Cores]")
            accelerator = "mps"
            dtype = float16

    # return any accelerator found
    return accelerator, dtype
