### TEST OF CONVOLUTIONAL SIREN ###
### idea inspired by hotgrits from EAI discord server, code is mine this time (wow!) ###

from math import log2

import torch

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from .siren import SirenLayer

# idea 1: standard conv siren with no bottleneck, scale-up #

#According to inceptionnet papers, greater efficiency comes from factorizing convolutions.
#Let's do that here and see if that helps lmao
#(https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202)
class FacConv(nn.Module):
	def __init__(self, in_channels, out_channels, k_size):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = (1, k_size))
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = (k_size, 1))
	def forward(self, x):
		return self.conv2(self.conv1(x))

#SIREN convolutional block structure:
#1. Convolutional layer
#2. Batch normalization
#3. SIREN network
class SIRENConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, image_size, w0=1., is_first=False, layer_activation=torch.sin, final_activation = None, multiply=None, learnable=False):
		super().__init__()
		if is_first:

		self.block = nn.Sequential(
			FacConv(in_channels=in_channels, out_channels=out_channels, k_size=kernel_size),
			nn.BatchNorm2d(out_channels),
			Rearrange('1 c h w -> (h w) c'),
			SirenLayer(dim_in=out_channels, dim_out=out_channels, w0=w0, is_first=is_first, layer_activation=layer_activation, final_activation=final_activation, multiply=multiply, learnable=learnable),
			Rearrange('(h w) c -> () c h w', h=image_size, w=image_size)
		)

	def forward(self, x):
		return self.block(x)

class SIRENConv(nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, image_size, num_scales=2, w0=30., layer_activation=torch.sin, final_activation=None, multiply=None, learnable=False):
		super().__init__()
		self.blocks = nn.ModuleList([])
		#sanity check
		assert (num_scales * 2) <= num_layers, 'Not enough layers to handle number of scales'
		assert log2(hidden_channels) % 1 == 0, 'hidden_channels must be a power of two'
		#Find the first hidden dimension to scale to
		hidden_channels_current = hidden_channels / (2 ** (num_scales - 1))

		#First layer
		self.blocks.append(SIRENConvBlock(
			in_channels = in_channels,
			out_channels = hidden_channels_current,
			kernel_size = 3, #temp for now
			image_size = image_size,
			w0 = w0,
			is_first = True,
			layer_activation = layer_activation,
			multiply = multiply,
			learnable = learnable
		))
		#Scale-up layers
		for _ in range(num_scales - 1):
			hidden_channels_next = hidden_channels_current * 2
			self.blocks.append(SIRENConvBlock(
				in_channels = hidden_channels_current,
				out_channels = hidden_channels_next,
				kernel_size = 3, #temp for now
				image_size = image_size,
				w0 = w0,
				layer_activation = layer_activation,
				multiply = multiply,
				learnable = learnable,
			))
			hidden_channels_current = hidden_channels_next

		#Intermediate layers (if there are any)
		if (num_layers - (2 * num_scales)) > 0:
			for _ in range(num_layers - (2 * num_scales)):
				self.blocks.append(SIRENConvBlock(
					in_channels = hidden_channels,
					out_channels = hidden_channels,
					kernel_size = 3, #temp for now
					image_size = image_size,
					w0 = w0,
					layer_activation = layer_activation,
					multiply = multiply,
					learnable = learnable,
				))

		#Scale-down layers
		hidden_channels_current = hidden_channels
		for _ in range(num_scales - 1):
			hidden_channels_next = hidden_channels_current / 2
			self.blocks.append(SIRENConvBlock(
				in_channels = hidden_channels_current,
				out_channels = hidden_channels_next,
				kernel_size = 3, #temp for now
				image_size = image_size,
				w0 = w0,
				layer_activation = layer_activation,
				multiply = multiply,
				learnable = learnable,
			))
			hidden_channels_current = hidden_channels_next

		#Last layer
		self.blocks.append(SIRENConvBlock(
			in_channels = hidden_channels_current,
			out_channels = out_channels,
			kernel_size = 3, #temp for now
			image_size = image_size,
			w0 = w0,
			final_activation = final_activation,
			multiply = multiply,
			learnable = learnable
		))

		self.image_size = image_size

	def forward(self, x):
		#Rearrange once to account for SirenWrapper's rearrange before feeding it into the network
		x = rearrange(x, '(h w) c -> () c h w', h = self.image_size, w = self.size)
		#Feed it into the network
		return self.blocks(x)