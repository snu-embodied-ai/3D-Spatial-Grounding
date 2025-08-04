import torch

import contextlib


# https://github.com/embodied-generalist/embodied-generalist/tree/567f2d809cf8b21289d7fb9d3b616ae9c405cba3
def maybe_autocast(model, dtype='bf16', enabled=True):
    # if on cpu, don't use autocast
    # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
    enable_autocast = model.device != torch.device('cpu')

    if dtype == 'bf16':
        dtype = torch.bfloat16
    elif dtype == 'fp16':
        dtype = torch.float16
    else:
        dtype = torch.float32

    if enable_autocast:
        return torch.amp.autocast('cuda', dtype=dtype, enabled=enabled)
    else:
        return contextlib.nullcontext()