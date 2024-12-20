import torch
import numpy as np
import math

from torch import Tensor
from torch import nn

from typing import Literal, Callable, Any

from torch.nn import functional as F


class RBConv(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 in_dims: tuple,
                 v_kernel_size: int,
                 h_kernel_size: int,
                 v_stride: int = 1,
                 h_stride: int = 1,
                 h_dilation: int = 1,
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
            h_padding = 0
            h_pad_mode = 'zeros'
        else:
            # Conv2D only allows for the same amount of padding on both sides
            h_padding = [required_same_padding(in_dims[i], h_kernel_size, h_stride, split=True)[1] for i in [0, 1]]
        
        out_height = conv_output_size(in_dims[-1], v_kernel_size, v_stride, dilation=1, pad=v_pad_mode!='valid')
        
        conv2d_in_channels = out_height*v_kernel_size*in_channels # concatenated neighborhoods
        conv2d_out_channels = out_height*out_channels

        self.conv2d = nn.Conv2d(in_channels=conv2d_in_channels, 
                                   out_channels=conv2d_out_channels, 
                                   kernel_size=h_kernel_size, 
                                   padding=tuple(h_padding), 
                                   stride=h_stride, 
                                   dilation=h_dilation,
                                   padding_mode=h_pad_mode,
                                   groups=out_height, 
                                   bias=bias,
                                   **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv2d_in_channels = conv2d_in_channels
        self.conv2d_out_channels = conv2d_out_channels
        
        self.in_height = in_dims[-1]
        self.out_height = out_height
        
        self.in_dims = in_dims
        # Conv2D only allows for the same amount of padding on both sides
        self.out_dims = [conv_output_size(in_dims[i], h_kernel_size, h_stride, dilation=h_dilation, 
                                             pad=h_pad_mode!='valid', equal_pad=True) for i in [0, 1]] + [out_height]
        
        self.v_pad = v_pad_mode!='valid'
        self.v_stride = v_stride
        self.v_kernel_size = v_kernel_size
        
        
    def forward(self, input: Tensor) -> Tensor:
        """
        tensor of shape [batch, inHeight*channels, width, depth]
        -> tensor of shape [batch, outHeight*channels, width, depth]
        """
        
        concatenated_neighborhoods = self._concat_vertical_neighborhoods(input)
        return self.conv2d.forward(concatenated_neighborhoods)
        
        
    def _concat_vertical_neighborhoods(self, tensor: Tensor) -> Tensor:
        """tensor of shape [batch, inHeight*channels, width, depth]
        -> [batch, outHeight*ksize*channels, width, depth]"""
        tensor = tensor.reshape(-1, self.in_height, self.in_channels, *self.in_dims[:2]) # split height and field dimension

        if self.v_pad:
            # pad height
            padding = required_same_padding(in_size=self.in_height, kernel_size=self.v_kernel_size, stride=self.v_stride, split=True)
            tensor = F.pad(tensor, (*([0,0]*3), *padding)) # shape:(b,padH,c,w,d)
        
        # compute neighborhoods
        tensor = tensor.unfold(dimension=1, size=self.v_kernel_size, step=self.v_stride) # shape:(b,outH,c,w,d,ksize)
        
        # concatenate neighboroods
        tensor = tensor.permute(0, 1, 5, 2, 3, 4) # shape:(b,outH,ksize,c,w,d)
        tensor = tensor.flatten(start_dim=1, end_dim=3) # shape:(b,outH*ksize*c,w,d)
        
        return tensor
    
    
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
        tensor = input.reshape(-1, self.in_height, self.in_channels, *self.in_dims[:2])
        tensor = tensor.permute(0, 2, 3, 1, 4)
        
        pooled_tensor = self.pool_op(tensor, [self.h_kernel_size, self.v_kernel_size, self.h_kernel_size])
        
        pooled_tensor = pooled_tensor.permute(0, 3, 1, 2, 4)
        batch, out_height, channels, out_width, out_depth = pooled_tensor.shape
        pooled_tensor = pooled_tensor.reshape(batch, out_height*channels, out_width, out_depth)
        
        return pooled_tensor    
    

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
        tensor = input.reshape(-1, self.in_height, self.in_channels, *self.in_dims[:2])
        tensor = tensor.permute(0, 2, 3, 1, 4)
        
        upsampled_tensor = F.interpolate(tensor, scale_factor=[self.h_scale, self.v_scale, self.h_scale], mode='trilinear')
        
        upsampled_tensor = upsampled_tensor.permute(0, 3, 1, 2, 4)
        batch, out_height, fieldsizes, out_width, out_depth = upsampled_tensor.shape
        upsampled_tensor = upsampled_tensor.reshape(batch, out_height*fieldsizes, out_width, out_depth)
        
        return upsampled_tensor