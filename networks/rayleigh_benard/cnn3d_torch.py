import torch
import numpy as np
import math

from torch import Tensor
from torch import nn

from typing import Literal

from torch.nn import functional as F


class RB3DConv(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 in_dims: tuple,
                 v_kernel_size: int,
                 h_kernel_size: int,
                 v_stride: int = 1,
                 h_stride: int = 1,
                 h_dilation: int = 1,
                 v_dilation: int = 1,
                 v_pad_mode: Literal['valid', 'zeros'] = 'zeros', 
                 h_pad_mode: Literal['valid', 'zeros', 'circular', 'reflect', 'replicate'] = 'circular',
                 bias: bool = True,
                 **kwargs
                 ):
        super().__init__()
        
        assert len(in_dims) == 3
        
        v_pad_mode = v_pad_mode.lower()
        h_pad_mode = h_pad_mode.lower()
        assert v_pad_mode.lower() in ['valid', 'zeros']
        assert h_pad_mode.lower() in ['valid', 'zeros', 'circular', 'reflect', 'replicate']
        
        if h_pad_mode == 'valid':
            h_padding = (0, 0)
            h_pad_mode = 'zeros'
        else:
            # Conv2D only allows for the same amount of padding on both sides
            h_padding = [required_same_padding(in_dims[i], h_kernel_size, h_stride, split=True)[1] for i in [0, 1]]
            
        self.v_padding = 0
        if v_pad_mode != 'valid':
            self.v_padding = required_same_padding(in_dims[2], v_kernel_size, v_stride, split=True)
        
        out_height = conv_output_size(in_dims[-1], v_kernel_size, v_stride, dilation=v_dilation, pad=v_pad_mode!='valid')

        self.conv2d = nn.Conv3d(in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=(h_kernel_size, h_kernel_size, v_kernel_size),
                                padding=(*h_padding, 0),  # vertical padding is done separately
                                stride=(h_stride, h_stride, 1), 
                                dilation=(h_dilation, h_dilation, v_dilation),
                                padding_mode=h_pad_mode,
                                bias=bias,
                                **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.in_height = in_dims[-1]
        self.out_height = out_height
        
        self.in_dims = in_dims
        # Conv2D only allows for the same amount of padding on both sides
        self.out_dims = [conv_output_size(in_dims[i], h_kernel_size, h_stride, dilation=h_dilation, 
                                             pad=h_pad_mode!='valid', equal_pad=True) for i in [0, 1]] + [out_height]
        
        
    def forward(self, input: Tensor) -> Tensor:
        """
        tensor of shape [batch, channels, width, depth, inHeight]
        -> tensor of shape [batch, channels, width, depth, outHeight]
        """
        # reshape to: [batch, channels, width, depth, height]
        # input = input.reshape(-1, self.in_height, self.in_channels, *self.in_dims[:2]).permute(0, 2, 3, 4, 1)
        
        # vertical padding (horizontal padding is done by conv layer)
        input = F.pad(input, self.v_padding, 'constant', 0)
        
        return self.conv2d.forward(input)
        
        
    def train(self, *args, **kwargs):
        return self.conv2d.train(*args, **kwargs)
    
    
    def eval(self, *args, **kwargs):
        return self.conv2d.eval(*args, **kwargs)
    
        
def conv_output_size(in_size: int, kernel_size: int, stride: int, dilation: int, pad: bool, equal_pad: bool = False) -> int:
    padding = 0
    if pad:
        pad_split = required_same_padding(in_size, kernel_size, stride, split=True)
        padding = 2*pad_split[1] if equal_pad else sum(pad_split)

    return ((in_size - dilation*(kernel_size-1) + padding - 1) // stride) + 1


def required_same_padding(in_size: int, kernel_size: int, stride: int, split: bool = False) -> int | tuple[int, int]:
    out_size = math.ceil(in_size/stride)
    padding = max((out_size-1) * stride - in_size + kernel_size, 0)
    
    if split:
        return math.floor(padding/2), math.ceil(padding/2)
    else:
        return padding
    

class RBPooling(nn.Module):
    def __init__(self, in_channels: int, in_dims: tuple, v_kernel_size: int, h_kernel_size: int, type: Literal['max', 'mean'] = 'max'):
        super().__init__()
        
        self.in_dims = in_dims
        self.out_dims = [in_dims[i] // h_kernel_size for i in [0, 1]] + [in_dims[-1] // v_kernel_size]
        
        self.in_height = in_dims[-1]
        self.out_height = self.out_dims[-1]
        
        self.in_channels = in_channels
        self.out_channels = in_channels
        
        self.v_kernel_size = v_kernel_size
        self.h_kernel_size = h_kernel_size
        
        self.pool_op = F.max_pool3d if type.lower() == 'max' else F.mean_pool3d
        
        
    def forward(self, input: Tensor) -> Tensor:                
        return self.pool_op(input, [self.h_kernel_size, self.h_kernel_size, self.v_kernel_size])         
    

class RBUpsampling(nn.Module):
    def __init__(self, in_channels: int, in_dims: tuple, v_scale: int, h_scale: int):
        super().__init__()
        
        self.in_dims = in_dims
        self.out_dims = [in_dims[i] * h_scale for i in [0, 1]] + [in_dims[-1] * v_scale]
        
        self.in_height = in_dims[-1]
        self.out_height = self.out_dims[-1]
        
        self.in_channels = in_channels
        self.out_channels = in_channels
        
        self.v_scale = v_scale
        self.h_scale = h_scale
        
        
    def forward(self, input: Tensor) -> Tensor:
        return F.interpolate(input, scale_factor=[self.h_scale, self.h_scale, self.v_scale], mode='trilinear')