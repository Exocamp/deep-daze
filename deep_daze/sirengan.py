import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn

from .utils import exists

#using nielsrolf's n-dim SIREN to allow for flexible output shapes
#https://github.com/lucidrains/siren-pytorch/pull/4
def get_grid(output_shape, min_val=-1, max_val=1):
    tensors = [torch.linspace(min_val, max_val, steps = i) for i in output_shape]
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, len(output_shape))
    return mgrid

#Activation layer - just Sine for now, we're keeping things simple until this actually works
class SineActivation(nn.Module):
	def __init__(self, w0 = 1.):
		super().__init__()
		self.w0 = w0

	def forward(self, x):
		return torch.sin(self.w0 * x)

class SirenLayer(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, final_activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = enable(use_bias, torch.zeros(dim_out))
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = enable(use_bias, nn.Parameter(bias))
        self.activation = SineActivation(w0=w0) if final_activation is None else final_activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
    	out = F.linear(x, self.weight, self.bias)
    	out = self.activation(out)

        return out

#SIREN network
class SirenNetwork(nn.Module):
	def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None):
		super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])

        self.layers.append(SirenLayer(
            dim_in = dim_in,
            dim_out = dim_hidden,
            w0 = w0_initial,
            use_bias = use_bias,
            is_first = True
        ))

        for ind in range(num_layers - 1):
            self.layers.append(SirenLayer(
                dim_in = dim_hidden,
                dim_out = dim_hidden,
                w0 = w0,
                use_bias = use_bias
            ))

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = SirenLayer(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, final_activation = final_activation)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.last_layer(x)

###############
#  GENERATOR  #
###############
class SirenG(nn.Module):
    def __init__(self, net, output_shape, latent_dim = None):
        super().__init__()
        assert isinstance(net, SirenNetwork), 'SirenWrapper must receive a Siren network'

        self.net = net
        self.output_shape = list(output_shape)[:-1]
        self.output_channels = list(output_shape)[-1]

        mgrid = get_grid(self.output_shape)
        self.register_buffer('grid', mgrid)

    def forward(self, target = None, *, latent = None, coords = None, output_shape = None):

        if coords is None:
            coords = self.grid.clone().detach().requires_grad_()

        out = self.net(coords)
        
        if output_shape is None:
            output_shape = self.output_shape

        #out = out.reshape(output_shape + [self.output_channels])
        out = out[None, :, :, :].permute(0, 3, 1, 2)

        if exists(target):
            return F.mse_loss(target, out)

        return out


###################
#  DISCRIMINATOR  #
###################

class SirenD(nn.Module):
    def __init__(self, net, output_shape, latent_dim = None):
        super().__init__()
        assert isinstance(net, SirenNetwork), 'SirenWrapper must receive a Siren network'

        self.net = net
        self.output_shape = output_shape
        print(f"output shape in d: {output_shape}")

    def forward(self, target = None, *, latent = None, coords = None, output_shape = None):
        assert exists(coords), "Discriminator needs an image!"

        out = self.net(coords)
        
        if output_shape is None:
            output_shape = self.output_shape

        out = out.reshape(output_shape)

        if exists(target):
            return F.mse_loss(target, out)

        return out

##################
#  EMBED FITTER  #
##################

class SirenE(nn.Module):
    def __init__(self, net, output_shape, latent_dim = None):
        super().__init__()
        assert isinstance(net, SirenNetwork), 'SirenWrapper must receive a Siren network'

        self.net = net
        self.output_shape = output_shape

        mgrid = get_grid(output_shape)
        self.register_buffer('grid', mgrid)

    def forward(self, target = None, *, latent = None, coords = None, output_shape = None):

        if coords is None:
            coords = self.grid.clone().detach().requires_grad_()

        out = self.net(coords)
        
        if output_shape is None:
            output_shape = self.output_shape

        out = out.reshape(output_shape)

        if exists(target):
            return F.mse_loss(target, out)

        return out

