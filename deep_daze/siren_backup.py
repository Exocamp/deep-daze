##Working further with Siren, to experiment deeper with the network and see what really clicks.
##Most of this taken from the siren-pytorch library, because I can't actually code lmao

import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from .utils import exists, enable

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

#Fourier features to be used on the input layer. thanks again alstro
#May need to adjust std for optimal performance
class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)

#Custom activation. Will it work? ¯\_(ツ)_/¯

class LayerActivation(nn.Module):
    def __init__(self, torch_activation=torch.sin, w0 = 1.):
        super().__init__()
        self.w0 = w0
        self.activation = torch_activation
    def forward(self, x):
        return self.activation(self.w0 * x)

#aight I guess I have to just import the whole Siren module. okay then.

class SirenLayer(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, layer_activation=torch.sin, final_activation = None, num_linears=1, multiply=None, erf_init=False):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.num_linears = num_linears
        self.multiply = multiply
        self.erf_init = erf_init

        weight = torch.zeros(dim_out, dim_in)
        bias = enable(use_bias, torch.zeros(dim_out))
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = enable(use_bias, nn.Parameter(bias))
        self.activation = LayerActivation(torch_activation=layer_activation, w0=w0) if final_activation is None else final_activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)
        #Apply erf onto weights and biases. This somehow makes it better. No I don't know why shut up
        if self.erf_init:
            weight.erf_()

        if exists(bias):
            bias.uniform_(-w_std, w_std)
            if self.erf_init:
                bias.erf_()

    def forward(self, x):
        for _ in range(self.num_linears):
            out = F.linear(x, self.weight, self.bias)
            if exists(self.multiply):
                out *= self.multiply
            out = self.activation(out)

        return out

#because I don't wanna do 2 repos, here's a more "open" SirenNet class, and by that I mean just changing activations on the layers themselves lol
class SirenNetwork(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, layer_activation = None, final_activation = None, num_linears = 1, multiply=None, fourier=True, erf_init=False):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])

        #Fourier
        if fourier:
          self.layers.append(FourierFeatures(
            in_features = dim_in,
            out_features = dim_hidden
          ))
        else:
          self.layers.append(SirenLayer(
            dim_in = dim_in,
            dim_out = dim_hidden,
            w0 = w0_initial,
            use_bias = use_bias,
            is_first = True,
            layer_activation = None if not exists(layer_activation) else LayerActivation(torch_activation=layer_activation),
            num_linears=num_linears,
            erf_init=erf_init
          ))

        for ind in range(num_layers - 1):
            self.layers.append(SirenLayer(
                dim_in = dim_hidden,
                dim_out = dim_hidden,
                w0 = w0,
                use_bias = use_bias,
                layer_activation = None if not exists(layer_activation) else LayerActivation(torch_activation=layer_activation),
                num_linears=num_linears
            ))
        
        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = SirenLayer(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, final_activation = final_activation, multiply=multiply)

    def forward(self, x, mods = None):
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, 'd -> () d')

        return self.last_layer(x)


class SirenWrapper(nn.Module):
    def __init__(self, net, image_width, image_height, latent_dim = None):
        super().__init__()
        assert isinstance(net, SirenNetwork), 'SirenWrapper must receive a Siren network'

        self.net = net
        self.image_width = image_width
        self.image_height = image_height

        self.modulator = None
        if exists(latent_dim):
            self.modulator = Modulator(
                dim_in = latent_dim,
                dim_hidden = net.dim_hidden,
                num_layers = net.num_layers
            )

        tensors = [torch.linspace(-1, 1, steps = image_width), torch.linspace(-1, 1, steps = image_height)]
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = rearrange(mgrid, 'h w c -> (h w) c')

        self.register_buffer('grid', mgrid)

    def forward(self, img = None, *, latent = None):
        modulate = exists(self.modulator)
        assert not (modulate ^ exists(latent)), 'latent vector must be only supplied if `latent_dim` was passed in on instantiation'

        mods = self.modulator(latent) if modulate else None

        coords = self.grid.clone().detach().requires_grad_()
        out = self.net(coords, mods)
        out = rearrange(out, '(h w) c -> () c h w', h = self.image_height, w = self.image_width)

        if exists(img):
            return F.mse_loss(img, out)

        return out