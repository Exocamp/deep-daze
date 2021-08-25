#Utility functions

import math

import torch
from torch import nn
from torch.nn import functional as F

def exists(val):
    return val is not None

#Clean up code.
def enable(condition, value):
    return value if condition else None

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

#clamp_with_grad

class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply

def unmap_pixels(x, logit_laplace_eps=0.25):
    return clamp_with_grad((x - logit_laplace_eps) / (1 - 2 * logit_laplace_eps), 0, 1)

#Very big dictionaries of torch functions. Use this as a one-stop shop to referencing everything related to them.
POINTWISE_FUNCS = {"sin": torch.sin, "cos": torch.cos, "tan": torch.tan,
"sinh": torch.sinh, "cosh": torch.cosh, "tanh": torch.tanh,
"sinc": torch.sinc, "asin": torch.asin, "atan": torch.atan, "acos": torch.acos, "erf": torch.erf}

ACTIVATION_FUNCS = {"identity": nn.Identity(), "sigmoid": nn.Sigmoid(), "relu": nn.ReLU(),
"leakyrelu": nn.LeakyReLU(negative_slope=0.2), "tanh": nn.Tanh(), "silu": nn.SiLU(),
"gelu": nn.GELU(), "celu": nn.CELU(), "selu": nn.SELU(), "elu": nn.ELU(), "softsign": nn.Softsign(),
"softplus": nn.Softplus(), "mish": nn.Mish()}
