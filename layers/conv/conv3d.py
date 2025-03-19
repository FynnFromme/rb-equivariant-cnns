from torch import Tensor
from torch import nn
from torch.nn import functional as F

from layers.conv import conv_utils
from typing import Literal


class RB3DConv(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 in_dims: tuple,
                 v_kernel_size: int,
                 h_kernel_size: int,
                 v_stride: int = 1,
                 h_stride: int = 1,
                 v_dilation: int = 1,
                 h_dilation: int = 1,
                 v_pad_mode: Literal['valid', 'zeros'] = 'zeros', 
                 h_pad_mode: Literal['valid', 'zeros', 'circular', 'reflect', 'replicate'] = 'circular',
                 bias: bool = True,
                 **kwargs):
        """A Rayleigh-Bénard (RB) 3D convolution wraps the standard 3D convolution (with vertical parameter
        sharing) to match the interface of the other layers without vertical parameter sharing.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            in_dims (tuple): The spatial dimensions of the input data.
            v_kernel_size (int): The vertical kernel size.
            h_kernel_size (int): The horizontal kernel size (in both directions).
            v_stride (int, optional): The vertical stride. Defaults to 1.
            h_stride (int, optional): The horizontal stride (in both directions). Defaults to 1.
            v_dilation (int, optional): The vertical dilation. Defaults to 1.
            h_dilation (int, optional): The horizontal dilation. Defaults to 1.
            v_pad_mode (str, optional): The padding applied to the vertical dimension. Must be either 'valid'
                for no padding or 'zero' for same padding with zeros. Defaults to 'zero'.
            h_pad_mode (str, optional): The padding applied to the horizontal dimensions. Must be one of the
                following: 'valid', 'zero', 'circular', 'reflect', 'replicate'. Defaults to 'circular'.
            bias (bool, optional): Whether to apply a bias to the layer's output. Defaults to True.
        """
        super().__init__()
        
        v_pad_mode = v_pad_mode.lower()
        h_pad_mode = h_pad_mode.lower()
        assert v_pad_mode in ['valid', 'zeros']
        assert h_pad_mode in ['valid', 'zeros', 'circular', 'reflect', 'replicate']
        assert len(in_dims) == 3
        
        if h_pad_mode == 'valid':
            h_padding = (0, 0)
            h_pad_mode = 'zeros'
        else:
            # Conv3D only allows for the same amount of padding on both sides
            h_padding = [conv_utils.required_same_padding(in_dims[i], h_kernel_size, h_stride, h_dilation, split=True)[1] 
                         for i in [0, 1]]
            
        self.v_padding = 0, 0
        if v_pad_mode != 'valid':
            self.v_padding = conv_utils.required_same_padding(in_dims[2], v_kernel_size, v_stride, 
                                                         dilation=v_dilation, split=True)
        
        out_height = conv_utils.conv_output_size(in_dims[-1], v_kernel_size, v_stride, dilation=v_dilation, 
                                                 pad=v_pad_mode!='valid')

        self.conv3d = nn.Conv3d(in_channels=in_channels, 
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
        
        self.in_dims = in_dims
        self.out_dims = [conv_utils.conv_output_size(in_dims[i], h_kernel_size, h_stride, dilation=h_dilation, 
                                                     pad=h_pad_mode!='valid', equal_pad=True) 
                         for i in [0, 1]] + [out_height]
        
        
    def forward(self, input: Tensor) -> Tensor:
        """Applies the convolution to a input tensor of shape [batch, inChannels, inWidth, inDepth, inHeight] 
        and results in a output tensor of shape [batch, outChannels, outWidth, outDepth, outHeight].

        Args:
            input (Tensor): The tensor to which the convolution is applied.

        Returns:
            Tensor: The output of the convolution.
        """
        # vertical padding (horizontal padding is done by conv operation)
        input = F.pad(input, self.v_padding, 'constant', 0)
        
        return self.conv3d.forward(input)
    

class RBPooling(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 in_dims: tuple, 
                 v_kernel_size: int, 
                 h_kernel_size: int, 
                 type: Literal['max', 'mean'] = 'max'):
        """The RB Pooling layer applies 3D spatial pooling.

        Args:
            in_channels (int): The number of input channels.
            in_dims (tuple): The spatial dimensions of the input data.
            v_kernel_size (int): The vertical pooling kernel size.
            h_kernel_size (int): The horizontal pooling kernel size (in both directions).
            type (str, optional): Whether to apply 'max' or 'mean' pooling. Defaults to 'max'.
        """
        super().__init__()
        
        self.in_dims = in_dims
        self.out_dims = [in_dims[i] // h_kernel_size for i in [0, 1]] + [in_dims[-1] // v_kernel_size]
        
        self.in_channels = in_channels
        self.out_channels = in_channels
        
        self.v_kernel_size = v_kernel_size
        self.h_kernel_size = h_kernel_size
        
        self.pool_op = F.max_pool3d if type.lower() == 'max' else F.avg_pool3d
        
        
    def forward(self, input: Tensor) -> Tensor:
        """Applies 3D spatial pooling to a tensor of shape [batch, channels, inWidth, inDepth, inHeight].

        Args:
            input (Tensor): The tensor to apply pooling to.

        Returns:
            Tensor: The pooled tensor of shape [batch, channels, outWidth, outDepth, outHeight]
        """
        return self.pool_op(input, [self.h_kernel_size, self.h_kernel_size, self.v_kernel_size])         
    

class RBUpsampling(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 in_dims: tuple, 
                 v_scale: int, 
                 h_scale: int):
        """The RB Upsampling layer applies 3D spatial upsampling.

        Args:
            in_channels (int): The number of input channels.
            in_dims (tuple): The spatial dimensions of the input data.
            v_scale (int): The vertical upsampling scale.
            h_scale (int): The horizontal upsampling scale (in both directions).
        """
        super().__init__()
        
        self.in_dims = in_dims
        self.out_dims = [in_dims[i] * h_scale for i in [0, 1]] + [in_dims[-1] * v_scale]
        
        self.in_channels = in_channels
        self.out_channels = in_channels
        
        self.v_scale = v_scale
        self.h_scale = h_scale
        
        
    def forward(self, input: Tensor) -> Tensor:
        """Applies 3D spatial upsampling to a tensor of shape [batch, channels, inWidth, inDepth, inHeight].

        Args:
            input (Tensor): The tensor to apply upsampling to.

        Returns:
            Tensor: The upsampled tensor of shape [batch, channels, outWidth, outDepth, outHeight]
        """   
        return F.interpolate(input, scale_factor=[self.h_scale, self.h_scale, self.v_scale], mode='trilinear')