#Code ported over from others, not mine. Credits to crowsonkb and alstroemeria313 for code

"""Good differentiable image resampling for PyTorch."""

from functools import update_wrapper
import math

import torch
from torch.nn import functional as F


def sinc(x):
    return torch.where(x != 0,  torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def odd(fn):
    return update_wrapper(lambda x: torch.sign(x) * fn(abs(x)), fn)


def _to_linear_srgb(input):
    cond = input <= 0.04045
    a = input / 12.92
    b = ((input + 0.055) / 1.055)**2.4
    return torch.where(cond, a, b)


def _to_nonlinear_srgb(input):
    cond = input <= 0.0031308
    a = 12.92 * input
    b = 1.055 * input**(1/2.4) - 0.055
    return torch.where(cond, a, b)


to_linear_srgb = odd(_to_linear_srgb)
to_nonlinear_srgb = odd(_to_nonlinear_srgb)


def resample(input, size, method, align_corners=True, is_srgb=False, mode='bicubic'):
    #methods: 'bigsleep', 'vqgan'
    assert method in [None, 'bigsleep', 'vqgan'], "Incorrect resample method"

    n, c, h, w = input.shape
    dh, dw = size
    num = 3 if method == 'bigsleep' else 2

    if is_srgb:
        input = to_linear_srgb(input)

    viewshape = [n * c, 1, h, w] if method == 'bigsleep' else [n, c, h, w]

    input = input.view(viewshape)

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, num), num).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, num), num).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    input = F.interpolate(input, size, mode=mode, align_corners=align_corners)

    if is_srgb:
        input = to_nonlinear_srgb(input)

    return input