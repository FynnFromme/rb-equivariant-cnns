import torch
import numpy as np

from prettytable import PrettyTable

from collections import OrderedDict
from torch import nn

from networks.rayleigh_benard.cnn_torch import RBConv, RBPooling, RBUpsampling

#? info: torch Conv2d uses he initialization

class _ConvBlock(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 in_dims: tuple,
                 v_kernel_size: int,
                 h_kernel_size: int,
                 input_drop_rate: float,
                 bias: bool = True,
                 nonlinearity: bool = True,
                 batch_norm: bool = True):
        
        layers = []
        
        conv = RBConv(in_channels=in_channels, 
                      out_channels=out_channels, 
                      in_dims=in_dims,
                      v_kernel_size=v_kernel_size, 
                      h_kernel_size=h_kernel_size,
                      bias=bias and not batch_norm, # bias has no effect when using batch norm
                      v_stride=1, h_stride=1,
                      v_pad_mode='zeros', h_pad_mode='circular')
        
        if input_drop_rate > 0: layers.append(nn.Dropout(p=input_drop_rate))
        layers.append(conv)
        if batch_norm: layers.append(nn.BatchNorm2d(conv.conv2d_out_channels))
        if nonlinearity: layers.append(nn.ELU())
        
        super().__init__(*layers)
        
        self.in_dims, self.out_dims = in_dims, conv.out_dims
        self.in_channels, self.out_channels = in_channels, out_channels
        

class RBAutoEncoder(nn.Sequential):
    def __init__(self, 
                 rb_dims: tuple,
                 encoder_channels: tuple,
                 latent_channels: int,
                 v_kernel_size: int = 3,
                 h_kernel_size: int = 3,
                 drop_rate: float = 0.2):
        layers = []
        self.out_shapes = OrderedDict()
        self.layer_params = OrderedDict()
        self.out_shapes['Input'] = [4, *rb_dims]
        
        # Encoder
        in_channels, in_dims = 4, rb_dims
        for i, out_channels in enumerate(encoder_channels, 1):
            layer_drop_rate = 0 if i == 1 else drop_rate
            
            layers.append(_ConvBlock(in_channels=in_channels, out_channels=out_channels, 
                                     in_dims=in_dims, v_kernel_size=v_kernel_size, h_kernel_size=h_kernel_size,
                                     input_drop_rate=layer_drop_rate, nonlinearity=True, batch_norm=True))
            in_channels = layers[-1].out_channels
            self.out_shapes[f'EncoderConv{i}'] = [out_channels, *in_dims]
            self.layer_params[f'EncoderConv{i}'] = _count_params(layers[-1])
            
            layers.append(RBPooling(in_channels=in_channels, in_dims=in_dims, v_kernel_size=2, h_kernel_size=2))
            in_dims = layers[-1].out_dims
            self.out_shapes[f'Pooling{i}'] = [out_channels, *in_dims]
            self.layer_params[f'Pooling{i}'] = _count_params(layers[-1])
           
        # Latent Space
        layers.append(_ConvBlock(in_channels=in_channels, out_channels=latent_channels, 
                                 in_dims=in_dims, v_kernel_size=v_kernel_size, h_kernel_size=h_kernel_size,
                                 input_drop_rate=drop_rate, nonlinearity=True, batch_norm=True))
        in_channels = layers[-1].out_channels
        self.out_shapes[f'LatentConv'] = [latent_channels, *in_dims]
        self.layer_params[f'LatentConv'] = _count_params(layers[-1])
        
        self.latent_shape = [latent_channels, *in_dims]
            
        # Decoder
        for i, out_channels in enumerate(reversed(encoder_channels), 1):            
            layers.append(_ConvBlock(in_channels=in_channels, out_channels=out_channels, 
                                     in_dims=in_dims, v_kernel_size=v_kernel_size, h_kernel_size=h_kernel_size,
                                     input_drop_rate=drop_rate, nonlinearity=True, batch_norm=True))
            in_channels = layers[-1].out_channels
            self.out_shapes[f'DecoderConv{i}'] = [out_channels, *in_dims]
            self.layer_params[f'DecoderConv{i}'] = _count_params(layers[-1])
            
            layers.append(RBUpsampling(in_channels=in_channels, in_dims=in_dims, v_scale=2, h_scale=2))
            in_dims = layers[-1].out_dims
            self.out_shapes[f'Upsampling{i}'] = [out_channels, *in_dims]
            self.layer_params[f'Upsampling{i}'] = _count_params(layers[-1])
        
        # Out Conv
        layers.append(_ConvBlock(in_channels=in_channels, out_channels=4, 
                                 in_dims=in_dims, v_kernel_size=v_kernel_size, h_kernel_size=h_kernel_size,
                                 input_drop_rate=drop_rate, nonlinearity=False, batch_norm=False))
        self.out_shapes['OutputConv'] = [4, *in_dims]
        self.layer_params['OutputConv'] = _count_params(layers[-1])
        
        self.in_dims, self.out_dims = tuple(layers[0].in_dims), tuple(layers[-1].out_dims)
        
        assert self.out_dims == self.in_dims == tuple(rb_dims)
        
        super().__init__(*layers)
       
        
    def summary(self):
        table = PrettyTable()
        table.field_names = ["Layer", 
                             "Output shape [c, |G|, w, d, h]", 
                             "Parameters"]
        table.align["Layer"] = "l"
        table.align["Output shape [c, |G|, w, d, h]"] = "r"
        table.align["Parameters"] = "r"
        
        for layer in self.out_shapes.keys():
            params = self.layer_params[layer] if layer in self.layer_params else 0
            table.add_row([layer, self.out_shapes[layer], f'{params:,}'])
            
        print(table)
            
        print(f'\nShape of latent space: {self.latent_shape}')
        
        print(f'\nLatent-Input-Ratio: {np.prod(self.latent_shape)/np.prod(self.out_shapes["Input"])*100:.2f}%')

        print(f'\nTrainable parameters: {_count_params(self):,}')
        
        
def _count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)