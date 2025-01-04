import numpy as np
from torch import nn
from torch import Tensor

import experiments.models.model_utils as model_utils
from collections import OrderedDict
from typing import Callable

from networks.cnn import RBConv, RBPooling, RBUpsampling


class _ConvBlock(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 in_dims: tuple,
                 v_kernel_size: int,
                 h_kernel_size: int,
                 input_drop_rate: float,
                 bias: bool = True,
                 nonlinearity: Callable = nn.ELU,
                 batch_norm: bool = True):
        """A convolution block (without vertical parameter sharing) with dropout, batch normalization
        and nonlinearity.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels
            in_dims (tuple): The spatial dimensions of the input data.
            v_kernel_size (int): The vertical kernel size.
            h_kernel_size (int): The horizontal kernel size (in both directions).
            input_drop_rate (float): The drop rate for dropout applied to the input of the conv block. Set to 0
                to turn off dropout.
            bias (bool, optional): Whether to apply a bias to the output of the convolution. 
                Bias is turned off automatically when using batch normalization as a bias has no effect when
                using batch normalization. Defaults to True.
            nonlinearity (Callable, optional): The nonlinearity applied to the conv output. Set to `None` to
                have no nonlinearity. Defaults to nn.ELU.
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to True.
        """
        
        conv = RBConv(in_channels=in_channels, 
                      out_channels=out_channels, 
                      in_dims=in_dims,
                      v_kernel_size=v_kernel_size, 
                      h_kernel_size=h_kernel_size,
                      bias=bias and not batch_norm, # bias has no effect when using batch norm
                      v_stride=1, h_stride=1,
                      v_pad_mode='zeros', h_pad_mode='circular')
        
        layers = []
        if input_drop_rate > 0: layers.append(nn.Dropout(p=input_drop_rate))
        layers.append(conv)
        if batch_norm: layers.append(nn.BatchNorm2d(conv.conv2d_out_channels))
        if nonlinearity: layers.append(nonlinearity())
        
        super().__init__(*layers)
        
        self.in_dims, self.out_dims = in_dims, conv.out_dims
        self.in_channels, self.out_channels = in_channels, out_channels
        

class RBAutoencoder(nn.Module):
    def __init__(self, 
                 rb_dims: tuple,
                 encoder_channels: tuple,
                 latent_channels: int,
                 v_kernel_size: int = 3,
                 h_kernel_size: int = 3,
                 drop_rate: float = 0.2,
                 nonlinearity: Callable = nn.ELU):
        """A Rayleigh-Bénard autoencoder based on 3D convolutions without vertical parameter sharing.

        Args:
            rb_dims (tuple): The spatial dimensions of the simulation data.
            encoder_channels (tuple): The channels of the encoder. Each entry results in a corresponding layer.
                The decoder uses the channels in reversed order.
            latent_channels (int): The number of channels in the latent space.
             v_kernel_size (int, optional): The vertical kernel size. Defaults to 3.
            h_kernel_size (int, optional): The horizontal kernel size (in both directions). Defaults to 3.
            drop_rate (float, optional): The drop rate used for dropout. Set to 0 to turn off dropout. 
                Defaults to 0.2.
            nonlinearity (Callable, optional): The nonlinearity applied to the conv output. Set to `None` to
                have no nonlinearity. Defaults to enn.ELU.
        """
        super().__init__()
        
        encoder_layers = []
        decoder_layers = []
        self.out_shapes = OrderedDict()
        self.layer_params = OrderedDict()
        self.out_shapes['Input'] = [4, *rb_dims]
        
        #####################
        ####   Encoder   ####
        #####################
        in_channels, in_dims = 4, rb_dims
        for i, out_channels in enumerate(encoder_channels, 1):
            layer_drop_rate = 0 if i == 1 else drop_rate # don't apply dropout to the networks input
            
            encoder_layers.append(_ConvBlock(in_channels=in_channels, 
                                             out_channels=out_channels, 
                                             in_dims=in_dims, 
                                             v_kernel_size=v_kernel_size, 
                                             h_kernel_size=h_kernel_size,
                                             input_drop_rate=layer_drop_rate, 
                                             nonlinearity=nonlinearity, 
                                             batch_norm=True))
            in_channels = encoder_layers[-1].out_channels
            self.out_shapes[f'EncoderConv{i}'] = [out_channels, *in_dims]
            self.layer_params[f'EncoderConv{i}'] = model_utils.count_trainable_params(encoder_layers[-1])
            
            encoder_layers.append(RBPooling(in_channels=in_channels, in_dims=in_dims, v_kernel_size=2, h_kernel_size=2))
            in_dims = encoder_layers[-1].out_dims
            self.out_shapes[f'Pooling{i}'] = [out_channels, *in_dims]
            self.layer_params[f'Pooling{i}'] = model_utils.count_trainable_params(encoder_layers[-1])
           
        ######################
        #### Latent Space ####
        ######################
        encoder_layers.append(_ConvBlock(in_channels=in_channels, 
                                         out_channels=latent_channels, 
                                         in_dims=in_dims, 
                                         v_kernel_size=v_kernel_size, 
                                         h_kernel_size=h_kernel_size,
                                         input_drop_rate=drop_rate, 
                                         nonlinearity=nonlinearity, 
                                         batch_norm=True))
        in_channels = encoder_layers[-1].out_channels
        self.out_shapes[f'LatentConv'] = [latent_channels, *in_dims]
        self.layer_params[f'LatentConv'] = model_utils.count_trainable_params(encoder_layers[-1])
        
        self.latent_shape = [latent_channels, *in_dims]
            
        #####################
        ####   Decoder   ####
        #####################
        for i, out_channels in enumerate(reversed(encoder_channels), 1):            
            decoder_layers.append(_ConvBlock(in_channels=in_channels, 
                                             out_channels=out_channels, 
                                             in_dims=in_dims, 
                                             v_kernel_size=v_kernel_size, 
                                             h_kernel_size=h_kernel_size,
                                             input_drop_rate=drop_rate, 
                                             nonlinearity=nonlinearity, 
                                             batch_norm=True))
            in_channels = decoder_layers[-1].out_channels
            self.out_shapes[f'DecoderConv{i}'] = [out_channels, *in_dims]
            self.layer_params[f'DecoderConv{i}'] = model_utils.count_trainable_params(decoder_layers[-1])
            
            decoder_layers.append(RBUpsampling(in_channels=in_channels, in_dims=in_dims, v_scale=2, h_scale=2))
            in_dims = decoder_layers[-1].out_dims
            self.out_shapes[f'Upsampling{i}'] = [out_channels, *in_dims]
            self.layer_params[f'Upsampling{i}'] = model_utils.count_trainable_params(decoder_layers[-1])
        
        ######################
        ####    Output    ####
        ######################
        decoder_layers.append(_ConvBlock(in_channels=in_channels, 
                                         out_channels=4, 
                                         in_dims=in_dims, 
                                         v_kernel_size=v_kernel_size, 
                                         h_kernel_size=h_kernel_size,
                                         input_drop_rate=drop_rate, 
                                         nonlinearity=None, 
                                         batch_norm=False))
        self.out_shapes['OutputConv'] = [4, *in_dims]
        self.layer_params['OutputConv'] = model_utils.count_trainable_params(decoder_layers[-1])
        
        self.in_dims, self.out_dims = tuple(encoder_layers[0].in_dims), tuple(decoder_layers[-1].out_dims)
        
        assert self.out_dims == self.in_dims == tuple(rb_dims)
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        
    
    def train(self, *args, **kwargs):
        """Sets module to training mode."""
        self.encoder.train(*args, **kwargs)
        self.decoder.train(*args, **kwargs)
    
    
    def eval(self, *args, **kwargs):
        """Sets module to evaluation mode."""
        self.encoder.eval(*args, **kwargs)
        self.decoder.eval(*args, **kwargs)
        
    
    def forward(self, input: Tensor) -> Tensor:
        """Forwards the input through the network and returns the output.

        Args:
            input (Tensor): The networks input of shape [batch, height*channels, width, depth]

        Returns:
            Tensor: The decoded output of shape [batch, height*channels, width, depth]
        """
        
        latent = self.encoder.forward(input)
        output = self.decoder.forward(latent)
        
        return output
        
    
    def encode(self, input: Tensor) -> Tensor:
        """Forwards the input through the encoder part and returns the latent representation.

        Args:
            input (Tensor): The networks input of shape [batch, height*channels, width, depth]

        Returns:
            Tensor: The latent representation of shape [batch, height*channels, width, depth]
        """
        return self.encoder.forward(input)
    
    
    def decode(self, latent: Tensor) -> Tensor:
        """Forwards the latent representation through the decoder part and returns the decoded output.

        Args:
            input (Tensor): The latent representation of shape [batch, height*channels, width, depth]

        Returns:
            Tensor: The decoded output of shape [batch, height*channels, width, depth]
        """
        return self.decoder.forward(latent)
       
        
    def summary(self):
        """Print summary of the model."""
        model_utils.summary(self, self.out_shapes, self.layer_params, self.latent_shape)