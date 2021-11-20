#Utility functions

import math

import torch
import torchvision.transforms as T

from torch import nn
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode

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

##AUGMENTS

#For some reason torchvision random affines and color jitter does not have a probability parameter
#Meaning that it happens 100% of the time. Bruh.
#so I gotta code that myself

bilinear = InterpolationMode.BILINEAR

class ActuallyRandomAffine(nn.Module):
    def __init__(self, degrees, translate=None, scale=None, shear=None, interpolation=bilinear, fill=0, fillcolor=None, resample=None, p=0.5):
        super().__init__()
        self.p = p
        self.t = T.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear, interpolation=interpolation, fill=fill, fillcolor=fillcolor, resample=None)

    def forward(self, img):
        if torch.rand(1) < self.p:
            return self.t(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p})"

class ActuallyRandomColorJitter(nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.5):
        super().__init__()
        self.p = p
        self.t = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def forward(self, img):
        if torch.rand(1) < self.p:
            return self.t(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p})"
