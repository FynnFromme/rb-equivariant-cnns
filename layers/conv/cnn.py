from torch import Tensor
from torch import nn
from torch.nn import functional as F

from layers.conv import conv_utils
from typing import Literal


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
                 **kwargs):
        """A Rayleigh-Bénard (RB) convolution uses convolutions with 3D kernels that are not
        shared vertically due to RB not being vertically translation equivariant.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            in_dims (tuple): The spatial dimensions of the input data.
            v_kernel_size (int): The vertical kernel size.
            h_kernel_size (int): The horizontal kernel size (in both directions).
            v_stride (int, optional): The vertical stride. Defaults to 1.
            h_stride (int, optional): The horizontal stride (in both directions). Defaults to 1.
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
        
        assert v_pad_mode.lower() in ['valid', 'zeros']
        assert h_pad_mode.lower() in ['valid', 'zeros', 'circular', 'reflect', 'replicate']
        assert len(in_dims) == 3
        
        if h_pad_mode == 'valid':
            h_padding = 0
            h_pad_mode = 'zeros'
        else:
            # Conv2D only allows for the same amount of padding on both sides
            h_padding = [network_utils.required_same_padding(in_dims[i], h_kernel_size, h_stride, h_dilation, split=True)[1] 
                         for i in [0, 1]]
        
        out_height = network_utils.conv_output_size(in_dims[-1], v_kernel_size, v_stride, 
                                            dilation=1, pad=v_pad_mode!='valid')
        
        # under the hood, this layer works by stacking the vertical neighborhoods of the input and then
        # applying a grouped 2d convolution
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
        
        # combined height and channel dimension
        self.conv2d_in_channels = conv2d_in_channels
        self.conv2d_out_channels = conv2d_out_channels
        
        self.in_height = in_dims[-1]
        self.out_height = out_height
        
        self.in_dims = in_dims
        self.out_dims = [network_utils.conv_output_size(in_dims[i], h_kernel_size, h_stride, dilation=h_dilation, 
                                                pad=h_pad_mode!='valid', equal_pad=True) 
                         for i in [0, 1]] + [out_height]
        
        self.v_pad = v_pad_mode!='valid'
        self.v_stride = v_stride
        self.v_kernel_size = v_kernel_size
        
        
    def forward(self, input: Tensor) -> Tensor:
        """Applies the convolution to a input tensor of shape [batch, inHeight*inChannels, 
        inWidth, inDepth] and results in a output tensor of shape [batch, outHeight*outChannels, 
        outWidth, outDepth].

        Args:
            input (Tensor): The tensor to which the convolution is applied.

        Returns:
            Tensor: The output of the convolution.
        """
        concatenated_neighborhoods = self._concat_vertical_neighborhoods(input)
        return self.conv2d.forward(concatenated_neighborhoods)
        
        
    def _concat_vertical_neighborhoods(self, tensor: Tensor) -> Tensor:
        """Concatenates the local vertical neighborhoods along the height/channel dimension.

        Args:
            tensor (Tensor): Input tensor of shape [batch, inHeight*channels, width, depth].

        Returns:
            Tensor: Output tensor of shape [batch, outHeight*ksize*channels, width, depth].
        """
        # split height and channel dimension
        tensor = tensor.reshape(-1, self.in_height, self.in_channels, *self.in_dims[:2]) 

        if self.v_pad:
            # pad height
            padding = network_utils.required_same_padding(self.in_height, self.v_kernel_size, 
                                                  self.v_stride, dilation=1, split=True)
            tensor = F.pad(tensor, (*([0,0]*3), *padding)) # shape (b,padH,c,w,d)
        
        # compute neighborhoods
        tensor = tensor.unfold(dimension=1, size=self.v_kernel_size, step=self.v_stride) # shape (b,outH,c,w,d,ksize)
        
        # concatenate neighboroods
        tensor = tensor.permute(0, 1, 5, 2, 3, 4) # shape (b,outH,ksize,c,w,d)
        tensor = tensor.flatten(start_dim=1, end_dim=3) # shape (b,outH*ksize*c,w,d)
        
        return tensor
    
    
    def train(self, *args, **kwargs):
        return self.conv2d.train(*args, **kwargs)
    
    
    def eval(self, *args, **kwargs):
        return self.conv2d.eval(*args, **kwargs)


class RBPooling(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 in_dims: tuple, 
                 v_kernel_size: int, 
                 h_kernel_size: int, 
                 type: Literal['max', 'mean'] = 'max'):
        """The RB Pooling layer applies 3D spatial pooling on the tensors with combined height 
        and channel dimensions received from the RBConv layer.

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
        
        self.in_height = in_dims[-1]
        self.out_height = self.out_dims[-1]
        
        self.in_channels = in_channels
        self.out_channels = in_channels
        
        self.v_kernel_size = v_kernel_size
        self.h_kernel_size = h_kernel_size
        
        self.pool_op = F.max_pool3d if type.lower() == 'max' else F.avg_pool3d
        
        
    def forward(self, input: Tensor) -> Tensor:
        """Applies 3D spatial pooling to a tensor of shape [batch, inHeight*channels, inWidth, inDepth].

        Args:
            input (Tensor): The tensor to apply pooling to.

        Returns:
            Tensor: The pooled tensor of shape [batch, outHeight*channels, outWidth, outDepth]
        """      
        # transform tensor to be able to apply the torch pooling operation
        tensor = input.reshape(-1, self.in_height, self.in_channels, *self.in_dims[:2])
        tensor = tensor.permute(0, 2, 3, 4, 1)
        
        # perform pooling
        pooled_tensor = self.pool_op(tensor, [self.h_kernel_size, self.h_kernel_size, self.v_kernel_size])
        
        # transform back into the original shape
        pooled_tensor = pooled_tensor.permute(0, 4, 1, 2, 3)
        batch, out_height, channels, out_width, out_depth = pooled_tensor.shape
        pooled_tensor = pooled_tensor.reshape(batch, out_height*channels, out_width, out_depth)
        
        return pooled_tensor    
    

class RBUpsampling(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 in_dims: tuple, 
                 v_scale: int, 
                 h_scale: int):
        """The RB Upsampling layer applies 3D spatial upsampling on the tensors with combined height 
        and channel dimensions received from the RBConv layer.

        Args:
            in_channels (int): The number of input channels.
            in_dims (tuple): The spatial dimensions of the input data.
            v_scale (int): The vertical upsampling scale.
            h_scale (int): The horizontal upsampling scale (in both directions).
        """
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
        """Applies 3D spatial upsampling to a tensor of shape [batch, inHeight*channels, inWidth, inDepth].

        Args:
            input (Tensor): The tensor to apply upsampling to.

        Returns:
            Tensor: The upsampled tensor of shape [batch, outHeight*channels, outWidth, outDepth]
        """     
        # transform tensor to be able to apply the torch upsampling operation
        tensor = input.reshape(-1, self.in_height, self.in_channels, *self.in_dims[:2])
        tensor = tensor.permute(0, 2, 3, 4, 1)
        
        # perform upsampling
        upsampled_tensor = F.interpolate(tensor, 
                                         scale_factor=[self.h_scale, self.h_scale, self.v_scale], 
                                         mode='trilinear')
        
        # transform back to original shape
        upsampled_tensor = upsampled_tensor.permute(0, 4, 1, 2, 3)
        batch, out_height, fieldsizes, out_width, out_depth = upsampled_tensor.shape
        upsampled_tensor = upsampled_tensor.reshape(batch, out_height*fieldsizes, out_width, out_depth)
        
        return upsampled_tensor