import numpy as np
import torch
from torch.nn import functional as F

from escnn import nn as enn
from escnn.nn import FieldType, GeometricTensor
from escnn.gspaces import GSpace
from escnn.group import Representation

from layers.conv import conv_utils
from typing import Literal, Callable, Any

from layers.conv.escnn_library_fix import fix_conv_eval
fix_conv_eval() # fixes "bug" in library


class RBSteerableConv(enn.EquivariantModule):
    def __init__(self, 
                 gspace: GSpace, 
                 in_fields: list[Representation], 
                 out_fields: list[Representation], 
                 in_dims: tuple,
                 v_kernel_size: int,
                 h_kernel_size: int,
                 v_stride: int = 1,
                 h_stride: int = 1,
                 h_dilation: int = 1,
                 v_pad_mode: Literal['valid', 'zero'] = 'zero', 
                 h_pad_mode: Literal['valid', 'zero', 'circular', 'reflect', 'replicate'] = 'circular',
                 bias: bool = True,
                 sigma: float | list[float] = None,
                 frequencies_cutoff: float | Callable[[float], int] = None,
                 rings: list[float] = None,
                 maximum_offset: int = None,
                 recompute: bool = False,
                 basis_filter: Callable[[dict], bool] = None,
                 initialize: bool = True,
                 **kwargs):
        """A Rayleigh-Bénard (RB) steerable convolution uses steerable convolutions with 3D kernels that are not
        shared vertically due to RB not being vertically translation equivariant.

        Args:
            gspace (GSpace): The group of transformations to be equivariant to. For `gspaces.flipRot2dOnR2(N)`
                the layer is equivariant to horizontal flips and rotations. Use `gspaces.rot2dOnR2(N)` for only
                rotational equivariance.
            in_fields (list[Representation]): The fields of the layer's input. This corresponds to input channels
                in standard convolutions.
            out_fields (list[Representation]): The fields of the layer's output. This corresponds to output 
                channels in standard convolutions.
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
            sigma (float | list[float], optional): Width of each ring where the bases are sampled. If only 
                one scalar is passed, it is used for all rings. Defaults to None.
            frequencies_cutoff (float | Callable[[float], int], optional): Function mapping the radii of the 
                    basis elements to the maximum frequency accepted. If a float values is passed, the maximum 
                    frequency is equal to the radius times this factor. Defaults to None, a more complex policy
                    is used.
            rings (list[float], optional): Radii of the rings where to sample the bases. Defaults to None.
            maximum_offset (int, optional): Number of additional (aliased) frequencies in the intertwiners for 
                finite groups. Defaults to None, all additional frequencies allowed by the frequencies cut-off
                are used.
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
        
        assert v_pad_mode.lower() in ['valid', 'zero']
        assert h_pad_mode.lower() in ['valid', 'zero', 'circular', 'reflect', 'replicate']
        assert len(in_dims) == 3
        
        if h_pad_mode == 'valid':
            h_padding = 0
            h_pad_mode = 'zero'
        else:
            # enn.R2Conv only allows for the same amount of padding on both sides
            h_padding = [conv_utils.required_same_padding(in_dims[i], h_kernel_size, h_stride, h_dilation, split=True)[1] 
                         for i in [0, 1]]
        
        out_height = conv_utils.conv_output_size(in_dims[-1], v_kernel_size, v_stride, 
                                            dilation=1, pad=v_pad_mode!='valid')
        
        # under the hood, this layer works by stacking the vertical neighborhoods of the input and then
        # applying a grouped 2d convolution
        r2_conv_in_type = FieldType(gspace, out_height*v_kernel_size*in_fields) # concatenated neighborhoods
        out_type = FieldType(gspace, out_height*out_fields)

        self.r2_conv = enn.R2Conv(in_type=r2_conv_in_type, 
                                       out_type=out_type, 
                                       kernel_size=h_kernel_size, 
                                       padding=tuple(h_padding), 
                                       stride=h_stride, 
                                       dilation=h_dilation,
                                       padding_mode=h_pad_mode,
                                       groups=out_height, 
                                       bias=bias,
                                       sigma=sigma,
                                       frequencies_cutoff=frequencies_cutoff,
                                       rings=rings,
                                       maximum_offset=maximum_offset,
                                       recompute=recompute,
                                       basis_filter=basis_filter,
                                       initialize=initialize,
                                       **kwargs)
        
        self.in_fields = in_fields
        self.out_fields = out_fields
        
        self.in_height = in_dims[-1]
        self.out_height = out_height
        
        # in_ and out_type are the stacked fields accross the height dimension
        self.in_type = FieldType(gspace, self.in_height*self.in_fields) # without any neighborhood concatenation
        self.r2_conv_in_type = r2_conv_in_type # with neighborhood concatenation
        self.out_type = out_type
        
        self.in_dims = in_dims
        self.out_dims = [conv_utils.conv_output_size(in_dims[i], h_kernel_size, h_stride, dilation=h_dilation, 
                                                pad=h_pad_mode!='valid', equal_pad=True) 
                         for i in [0, 1]] + [out_height]
        
        self.v_pad = v_pad_mode!='valid' # whether same padding is applied
        self.v_stride = v_stride
        self.v_kernel_size = v_kernel_size
        
        
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        """Applies the convolution to a geometric input tensor of shape [batch, inHeight*sum(inFieldsizes), 
        inWidth, inDepth] and results in a geometric output tensor of shape [batch, outHeight*sum(outFieldsizes), 
        outWidth, outDepth].

        Args:
            input (GeometricTensor): The tensor to which the convolution is applied.

        Returns:
            GeometricTensor: The output of the convolution.
        """
        assert input.type == self.in_type
        
        concatenated_neighborhoods = self._concat_vertical_neighborhoods(input)
        
        return self.r2_conv.forward(concatenated_neighborhoods)
        
        
    def _concat_vertical_neighborhoods(self, geom_tensor: GeometricTensor) -> GeometricTensor:
        """Concatenates the local vertical neighborhoods along the height/field dimension.

        Args:
            geom_tensor (GeometricTensor): Input tensor of shape [batch, inHeight*sum(fieldsizes), width, depth].

        Returns:
            GeometricTensor: Output tensor of shape [batch, outHeight*ksize*sum(fieldsizes), width, depth].
        """
        # split height and field dimension
        tensor = geom_tensor.tensor.reshape(-1, 
                                            self.in_height, 
                                            sum(field.size for field in self.in_fields), 
                                            *self.in_dims[:2])

        if self.v_pad:
            # pad height
            padding = conv_utils.required_same_padding(self.in_height, self.v_kernel_size, 
                                                  self.v_stride, dilation=1, split=True)
            tensor = F.pad(tensor, (*([0,0]*3), *padding)) # shape (b,padH,fields,w,d)
        
        # compute neighborhoods
        tensor = tensor.unfold(dimension=1, size=self.v_kernel_size, step=self.v_stride) # shape (b,outH,fields,w,d,ksize)
        
        # concatenate neighboroods
        tensor = tensor.permute(0, 1, 5, 2, 3, 4) # shape (b,outH,ksize,fields,w,d)
        tensor = tensor.flatten(start_dim=1, end_dim=3) # shape (b,outH*ksize*fields,w,d)
        
        return GeometricTensor(tensor, self.r2_conv_in_type)
    
    
    def evaluate_output_shape(self, input_shape: tuple) -> tuple:
        """Computes the shape of the output tensor.

        Args:
            input_shape (tuple): The shape of the input tensor.

        Returns:
            tuple: The corresponding shape of the output.
        """
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
    
        batch_size = input_shape[0]
        
        return (batch_size, self.out_type.size) + tuple(self.out_dims[:2])
    
    
    def train(self, *args, **kwargs):
        return self.r2_conv.train(*args, **kwargs)
    
    
    def eval(self, *args, **kwargs):
        return self.r2_conv.eval(*args, **kwargs)
    
    
    def check_equivariance(self, atol: float = 1e-7, rtol: float = 1e-5) -> list[tuple[Any, float]]:
        """Method that automatically tests the equivariance of the current module.
        
        Returns:
            list: A list containing containing for each testing element a pair with that element and 
            the corresponding equivariance error.
        """
        
        training = self.training
        self.eval()
    
        x = torch.randn(3, self.in_type.size, *self.in_dims[:2])
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
        """The RB Pooling layer applies 3D spatial pooling on the geometric tensors with combined height 
        and field dimensions received from the RBSteerableConv layer.

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
        
        self.in_type = FieldType(gspace, self.in_height * self.in_fields)
        self.out_type = FieldType(gspace, self.out_height * self.in_fields)
        
        self.pool_op = F.max_pool3d if type.lower() == 'max' else F.avg_pool3d
        
        
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        """Applies 3D spatial pooling to a geometric tensor of shape 
        [batch, inHeight*sum(fieldsizes), inWidth, inDepth].

        Args:
            input (GeometricTensor): The tensor to apply pooling to.

        Returns:
            GeometricTensor: The pooled tensor of shape [batch, outHeight*sum(fieldsizes), outWidth, outDepth]
        """
        assert input.type == self.in_type
        
        # transform tensor to be able to apply the torch pooling operation
        tensor = input.tensor.reshape(-1, 
                                      self.in_height, 
                                      sum(field.size for field in self.in_fields), 
                                      *self.in_dims[:2])
        tensor = tensor.permute(0, 2, 3, 1, 4)
        
        # perform pooling
        pooled_tensor = self.pool_op(tensor, [self.h_kernel_size, self.v_kernel_size, self.h_kernel_size])
        
        # transform back into the original shape
        pooled_tensor = pooled_tensor.permute(0, 3, 1, 2, 4)
        batch, out_height, fieldsizes, out_width, out_depth = pooled_tensor.shape
        pooled_tensor = pooled_tensor.reshape(batch, out_height*fieldsizes, out_width, out_depth)
        
        return GeometricTensor(pooled_tensor, FieldType(input.type.gspace, out_height*self.in_fields))
    
    
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
        
        out_width = in_width // self.h_kernel_size
        out_depth = in_depth // self.h_kernel_size
        
        return (batch, self.out_type.size, out_width, out_depth)


class RBUpsampling(enn.EquivariantModule):
    def __init__(self, 
                 gspace: GSpace, 
                 in_fields: list[Representation], 
                 in_dims: tuple, 
                 v_scale: int, 
                 h_scale: int):
        """The RB Upsampling layer applies 3D spatial upsampling on the geometric tensors with combined height 
        and field dimensions received from the RBSteerableConv layer.

        Args:
            gspace (GSpace): The gspace of the geometric tensor.
            in_fields (list[Representation]): The fields of the layer's input. This corresponds to input channels
                in standard convolutions.
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
        
        self.in_type = FieldType(gspace, self.in_height * self.in_fields)
        self.out_type = FieldType(gspace, self.out_height * self.in_fields)
        
        
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        """Applies 3D spatial upsampling to a geometric tensor of shape 
        [batch, inHeight*sum(fieldsizes), inWidth, inDepth].

        Args:
            input (GeometricTensor): The tensor to upsample.

        Returns:
            GeometricTensor: The upsampled tensor of shape [batch, outHeight*sum(fieldsizes), outWidth, outDepth]
        """
        assert input.type == self.in_type
        
        # transform tensor to be able to apply the torch upsampling operation
        tensor = input.tensor.reshape(-1, 
                                      self.in_height, 
                                      sum(field.size for field in self.in_fields), 
                                      *self.in_dims[:2])
        tensor = tensor.permute(0, 2, 3, 1, 4)
        
        # perform upsampling
        upsampled_tensor = F.interpolate(tensor, 
                                         scale_factor=[self.h_scale, self.v_scale, self.h_scale], 
                                         mode='trilinear')
        
        # transform back to original shape
        upsampled_tensor = upsampled_tensor.permute(0, 3, 1, 2, 4)
        batch, out_height, fieldsizes, out_width, out_depth = upsampled_tensor.shape
        upsampled_tensor = upsampled_tensor.reshape(batch, out_height*fieldsizes, out_width, out_depth)
        
        return GeometricTensor(upsampled_tensor, FieldType(input.type.gspace, out_height*self.in_fields))
    
    
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
        
        out_width = in_width * self.h_scale
        out_depth = in_depth * self.h_scale
        
        return (batch, self.out_type.size, out_width, out_depth)