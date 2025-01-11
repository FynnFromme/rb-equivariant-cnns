import numpy as np

import torch
from torch.nn import functional as F

from escnn import nn as enn
from escnn.nn import FieldType, GeometricTensor
from escnn.gspaces import GSpace
from escnn.group import Representation

from networks import network_utils
from typing import Literal, Callable, Any


class RBSteerable3DConv(enn.EquivariantModule):
    def __init__(self, 
                 gspace: GSpace, 
                 in_fields: list[Representation], 
                 out_fields: list[Representation],  
                 in_dims: tuple,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 v_pad_mode: Literal['valid', 'zeros'] = 'zeros', 
                 h_pad_mode: Literal['valid', 'zeros', 'circular', 'reflect', 'replicate'] = 'circular',
                 bias: bool = True,
                 sigma: float | list[float] = None,
                 frequencies_cutoff: float | Callable[[float], int] = None,
                 rings: list[float] = None,
                 recompute: bool = False,
                 basis_filter: Callable[[dict], bool] = None,
                 initialize: bool = True,
                 **kwargs):
        """A Rayleigh-Bénard (RB) 3D convolution wraps the standard 3D convolution (with vertical parameter
        sharing) to match the interface of the other layers without vertical parameter sharing.

        Args:
            gspace (GSpace): The group of transformations to be equivariant to. For 
                `utils.flipRot2dOnR3.flipRot2dOnR3(N)` the block is equivariant to horizontal flips 
                and rotations. Use `gspaces.rot2dOnR3(N)` for only rotational equivariance.
            in_fields (list[Representation]): The fields of the layer's input. This corresponds to input channels
                in standard convolutions.
            out_fields (list[Representation]): The fields of the layer's output. This corresponds to output 
                channels in standard convolutions.
            in_dims (tuple): The spatial dimensions of the input data.
            kernel_size (int): The kernel size (in all dimensions).
            stride (int, optional): The stride (in all dimensions). Defaults to 1.
            dilation (int, optional): The dilation (in all dimensions). Defaults to 1.
            v_pad_mode (str, optional): The padding applied to the vertical dimension. Must be either 'valid'
                for no padding or 'zeros' for same padding with zeros. Defaults to 'zeros'.
            h_pad_mode (str, optional): The padding applied to the horizontal dimensions. Must be one of the
                following: 'valid', 'zeros', 'circular', 'reflect', 'replicate'. Defaults to 'circular'.
            bias (bool, optional): Whether to apply a bias to the layer's output. Defaults to True.
            sigma (float | list[float], optional): Width of each ring where the bases are sampled. If only 
                one scalar is passed, it is used for all rings. Defaults to None.
            frequencies_cutoff (float | Callable[[float], int], optional): Function mapping the radii of the 
                    basis elements to the maximum frequency accepted. If a float values is passed, the maximum 
                    frequency is equal to the radius times this factor. Defaults to None, a more complex policy
                    is used.
            rings (list[float], optional): Radii of the rings where to sample the bases. Defaults to None.
            recompute (bool, optional): If True, recomputes a new basis for the equivariant kernels.
                Defaults to False, it  caches the basis built or reuse a cached one, if it is found.
            basis_filter (Callable[[dict], bool], optional): Function which takes as input a descriptor of a 
                basis element (as a dictionary) and returns a boolean value: whether to preserve (`True`) or
                discard (`False`) the basis element. Defaults to `None`, no filtering is applied.
            initialize (bool, optional): Whether to initialize the weights via he initialization. Defaults to True.
        """
        super().__init__()
        
        v_pad_mode = v_pad_mode.lower()
        h_pad_mode = h_pad_mode.lower()
        
        assert v_pad_mode.lower() in ['valid', 'zeros']
        assert h_pad_mode.lower() in ['valid', 'zeros', 'circular', 'reflect', 'replicate']
        assert len(in_dims) == 3
        
        if h_pad_mode == 'valid':
            h_padding = (0, 0)
            h_pad_mode = 'zeros'
        else:
            # Conv3D only allows for the same amount of padding on both sides
            h_padding = [network_utils.required_same_padding(in_dims[i], kernel_size, stride, dilation, split=True)[1] 
                         for i in [0, 1]]
            
        self.v_padding = 0, 0
        if v_pad_mode != 'valid':
            self.v_padding = network_utils.required_same_padding(in_dims[2], kernel_size, stride, 
                                                                 dilation=dilation, split=True)
        
        out_height = network_utils.conv_output_size(in_dims[-1], kernel_size, stride, dilation=dilation, 
                                                    pad=v_pad_mode!='valid')


        in_type = FieldType(gspace, in_fields)
        out_type = FieldType(gspace, out_fields)

        self.r3conv = enn.R3Conv(in_type=in_type, 
                                 out_type=out_type, 
                                 kernel_size=kernel_size, 
                                 padding=(0, *h_padding), 
                                 stride=stride, 
                                 dilation=dilation,
                                 padding_mode=h_pad_mode,
                                 bias=bias,
                                 sigma=sigma,
                                 frequencies_cutoff=frequencies_cutoff,
                                 rings=rings,
                                 recompute=recompute,
                                 basis_filter=basis_filter,
                                 initialize=initialize,
                                 **kwargs)
        
        self.in_type = in_type
        self.out_type = out_type
        
        self.in_height = in_dims[-1]
        self.out_height = out_height
        
        self.in_dims = in_dims
        self.out_dims = [network_utils.conv_output_size(in_dims[i], kernel_size, stride, dilation=dilation, 
                                                pad=h_pad_mode!='valid', equal_pad=True) 
                         for i in [0, 1]] + [out_height]
        
        
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        """Applies the convolution to a geometric input tensor of shape [batch, sum(inFieldsizes), inHeight, inWidth, 
        inDepth] and results in a geometric output tensor of shape [batch, sum(outFieldsizes), outHeight, outWidth, outDepth].

        Args:
            input (GeometricTensor): The tensor to which the convolution is applied.

        Returns:
            GeometricTensor: The output of the convolution.
        """
        assert input.type == self.in_type
         
        # vertical padding (horizontal padding is done by conv operation)
        input_tensor = input.tensor
        input_tensor = F.pad(input_tensor, (0, 0, 0, 0, *self.v_padding), 'constant', 0)
        input = GeometricTensor(input_tensor, self.in_type)
        
        return self.r3conv.forward(input)
        
        
    def train(self, *args, **kwargs):
        return self.r3conv.train(*args, **kwargs)
    
    
    def eval(self, *args, **kwargs):
        return self.r3conv.eval(*args, **kwargs)
    
    
    def evaluate_output_shape(self, input_shape: tuple) -> tuple:
        """Computes the shape of the output tensor.

        Args:
            input_shape (tuple): The shape of the input tensor.

        Returns:
            tuple: The corresponding shape of the output.
        """
        assert len(input_shape) == 5
        assert input_shape[1] == self.in_type.size
    
        batch_size = input_shape[0]
        
        return (batch_size, self.out_type.size) + (self.out_dims[0],) + tuple(self.out_dims[:2])
    
    
    def check_equivariance(self, atol: float = 1e-7, rtol: float = 1e-5) -> list[tuple[Any, float]]:
        """Method that automatically tests the equivariance of the current module.
        
        Returns:
            list: A list containing containing for each testing element a pair with that element and 
            the corresponding equivariance error.
        """
        
        training = self.training
        self.eval()
    
        x = torch.randn(3, self.in_type.size, self.in_dims[-1], *self.in_dims[:2])
        x = GeometricTensor(x, self.in_type)
        
        errors = []
        for el in self.out_type.testing_elements:            
            out1 = self(x).transform(el).tensor.detach().numpy()
            out2 = self(x.transform(el)).tensor.detach().numpy()
        
            errs = out1 - out2
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
        
            assert np.allclose(out1, out2, atol=atol, rtol=rtol), \
                f'The error found during equivariance check with element "{el}" \
                    is too high: max = {errs.max()}, mean = {errs.mean()} var ={errs.var()}'
            
            errors.append((el, errs.mean()))
            
        self.train(training)
        
        return errors
    

class RBPooling(enn.EquivariantModule):
    def __init__(self, 
                 gspace: GSpace, 
                 in_fields: list[Representation], 
                 in_dims: tuple, 
                 v_kernel_size: int, 
                 h_kernel_size: int, 
                 type: Literal['max', 'mean'] = 'max'):
        """The RB Pooling layer applies 3D spatial poolingon the geometric tensors received from the 
        RBSteerable3DConv layer.

        Args:
            gspace (GSpace): The gspace of the geometric tensor.
            in_fields (list[Representation]): The fields of the layer's input. This corresponds to input channels
                in standard convolutions.
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
        
        self.in_fields = in_fields
        self.out_fields = in_fields
        
        self.v_kernel_size = v_kernel_size
        self.h_kernel_size = h_kernel_size
        
        self.in_type = FieldType(gspace, self.in_fields)
        self.out_type = FieldType(gspace, self.in_fields)
        
        self.pool_op = F.max_pool3d if type.lower() == 'max' else F.mean_pool3d
        
        
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        """Applies 3D spatial pooling to a geometric tensor of shape 
        [batch, sum(fieldsizes), inHeight, inWidth, inDepth].

        Args:
            input (GeometricTensor): The tensor to apply pooling to.

        Returns:
            GeometricTensor: The pooled tensor of shape [batch, sum(fieldsizes), outHeight, outWidth, outDepth]
        """
        input_tensor = input.tensor
        output_tensor = self.pool_op(input_tensor, [self.v_kernel_size, self.h_kernel_size, self.h_kernel_size])
        output = GeometricTensor(output_tensor, self.out_type)
        return output
        
    
    
    def evaluate_output_shape(self, input_shape: tuple) -> tuple:
        """Computes the shape of the output tensor.

        Args:
            input_shape (tuple): The shape of the input tensor.

        Returns:
            tuple: The corresponding shape of the output.
        """
        assert len(input_shape) == 5
        assert input_shape[1] == self.in_type.size
        
        batch, _, in_width, in_depth = input_shape
        
        out_height = self.in_height // self.v_kernel_size
        out_width = in_width // self.h_kernel_size
        out_depth = in_depth // self.h_kernel_size
        
        return (batch, self.out_type.size, out_height, out_width, out_depth)     
    

class RBUpsampling(enn.EquivariantModule):
    def __init__(self,  
                 gspace: GSpace, 
                 in_fields: list[Representation], 
                 in_dims: tuple, 
                 v_scale: int, 
                 h_scale: int):
        """The RB Upsampling layer applies 3D spatial upsampling on the geometric tensors received from the 
        RBSteerable3DConv layer.

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
        
        self.in_fields = in_fields
        self.out_fields = in_fields
        
        self.v_scale = v_scale
        self.h_scale = h_scale
        
        self.in_type = FieldType(gspace, self.in_fields)
        self.out_type = FieldType(gspace, self.in_fields)
        
        
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        """Applies 3D spatial upsampling to a geometric tensor of shape 
        [batch, sum(fieldsizes), inHeight, inWidth, inDepth].

        Args:
            input (GeometricTensor): The tensor to upsample.

        Returns:
            GeometricTensor: The upsampled tensor of shape [batch, sum(fieldsizes), outHeight, outWidth, outDepth]
        """
        
        input_tensor = input.tensor
        output_tensor = F.interpolate(input_tensor, scale_factor=[self.v_scale, self.h_scale, self.h_scale], mode='trilinear')
        output = GeometricTensor(output_tensor, self.out_type)
        return output
    
    
    def evaluate_output_shape(self, input_shape: tuple) -> tuple:
        """Computes the shape of the output tensor.

        Args:
            input_shape (tuple): The shape of the input tensor.

        Returns:
            tuple: The corresponding shape of the output.
        """
        
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        
        batch, _, in_width, in_depth = input_shape
        
        out_height = self.in_height * self.v_scale
        out_width = in_width * self.h_scale
        out_depth = in_depth * self.h_scale
        
        return (batch, self.out_type.size, out_height, out_width, out_depth)