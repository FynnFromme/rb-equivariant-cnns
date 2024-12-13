import torch
import numpy as np
import math

from typing import Literal, Callable, Any

from escnn import nn as enn
from escnn.nn import FieldType, GeometricTensor
from escnn.gspaces import GSpace
from escnn.group import Representation

from torch.nn import functional as F


class RBSteerableConv(enn.EquivariantModule):
    def __init__(self, 
                 gspace: GSpace, 
                 in_fields: list[Representation], 
                 out_fields: list[Representation], 
                 in_dims: int,
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
                 **kwargs
                 ):
        super().__init__()
        
        assert len(in_dims) == 3
        
        v_pad_mode = v_pad_mode.lower()
        h_pad_mode = h_pad_mode.lower()
        assert v_pad_mode.lower() in ['valid', 'zero']
        assert h_pad_mode.lower() in ['valid', 'zero', 'circular', 'reflect', 'replicate']
        
        if h_pad_mode == 'valid':
            h_padding = 0
            h_pad_mode = 'zero'
        else:
            # enn.R2Conv only allows for the same amount of padding on both sides
            h_padding = [required_same_padding(in_dims[i], h_kernel_size, h_stride, split=True)[1] for i in [0, 1]]
        
        out_height = conv_output_size(in_dims[-1], v_kernel_size, v_stride, dilation=1, pad=v_pad_mode!='valid')
        
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
        
        self.in_type = FieldType(gspace, self.in_height*self.in_fields) # without any neighborhood concatenation
        self.r2_conv_in_type = r2_conv_in_type # with any neighborhood concatenation
        self.out_type = out_type
        
        self.in_dims = in_dims
        # enn.R2Conv only allows for the same amount of padding on both sides
        self.out_dims = [conv_output_size(in_dims[i], h_kernel_size, h_stride, dilation=h_dilation, 
                                             pad=h_pad_mode!='valid', equal_pad=True) for i in [0, 1]] + [out_height]
        
        self.v_pad = v_pad_mode!='valid'
        self.v_stride = v_stride
        self.v_kernel_size = v_kernel_size
        
        
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        """
        geomTensor of shape [batch, inHeight*sum(inFieldsizes), width, depth]
        -> geomTensor of shape [batch, outHeight*sum(outFieldsizes), width, depth]
        """
        assert input.type == self.in_type
        
        concatenated_neighborhoods = self._concat_vertical_neighborhoods(input)
        return self.r2_conv.forward(concatenated_neighborhoods)
        
        
    def _concat_vertical_neighborhoods(self, geom_tensor: GeometricTensor) -> GeometricTensor:
        """geomTensor of shape [batch, inHeight*sum(fieldsizes), width, depth]
        -> [batch, outHeight*ksize*sum(fieldsizes), width, depth]"""
        tensor = geom_tensor.tensor.reshape(-1, self.in_height, sum(field.size for field in self.in_fields), *self.in_dims[:2]) # split height and field dimension

        if self.v_pad:
            # pad height
            padding = required_same_padding(in_size=self.in_height, kernel_size=self.v_kernel_size, stride=self.v_stride, split=True)
            tensor = F.pad(tensor, (*([0,0]*3), *padding)) # shape:(b,padH,c*t,w,d)
        
        # compute neighborhoods
        tensor = tensor.unfold(dimension=1, size=self.v_kernel_size, step=self.v_stride) # shape:(b,outH,c*t,w,d,ksize)
        
        # concatenate neighboroods
        tensor = tensor.permute(0, 1, 5, 2, 3, 4) # shape:(b,outH,ksize,c*t,w,d)
        tensor = tensor.flatten(start_dim=1, end_dim=3) # shape:(b,outH*ksize*c*t,w,d)
        
        return GeometricTensor(tensor, self.r2_conv_in_type)
    
    
    def evaluate_output_shape(self, input_shape: tuple) -> tuple:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
    
        batch_size = input_shape[0]
        
        return (batch_size, self.out_type.size) + tuple(self.in_dims[:2])
    
    
    def train(self, *args, **kwargs):
        return self.r2_conv.train(*args, **kwargs)
    
    
    def check_equivariance(self, atol: float = 1e-7, rtol: float = 1e-5) -> list[tuple[Any, float]]:
        r"""
        
        Method that automatically tests the equivariance of the current module.
        The default implementation of this method relies on :meth:`escnn.nn.GeometricTensor.transform` and uses the
        the group elements in :attr:`~escnn.nn.FieldType.testing_elements`.
        
        This method can be overwritten for custom tests.
        
        Returns:
            a list containing containing for each testing element a pair with that element and the corresponding
            equivariance error
        
        """
        
        training = self.training
        self.eval()
    
        x = torch.randn(3, self.in_type.size, *self.in_dims[:2])
        x = GeometricTensor(x, self.in_type)
        
        errors = []
        for el in self.out_type.testing_elements:
            el = self.in_type.gspace.fibergroup.sample()
            print(el)
            
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
    

class RBMaxPool(enn.EquivariantModule):
    def __init__(self, gspace: GSpace, in_fields: list[Representation], in_dims: tuple, v_kernel_size: int, h_kernel_size: int):
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
        
        
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        assert input.type == self.in_type
        
        tensor = input.tensor.reshape(-1, self.in_height, sum(field.size for field in self.in_fields), *self.in_dims[:2])
        tensor = tensor.permute(0, 2, 3, 1, 4)
        
        pooled_tensor = F.max_pool3d(tensor, [self.h_kernel_size, self.v_kernel_size, self.h_kernel_size])
        
        pooled_tensor = pooled_tensor.permute(0, 3, 1, 2, 4)
        batch, out_height, fieldsizes, out_width, out_depth = pooled_tensor.shape
        pooled_tensor = pooled_tensor.reshape(batch, out_height*fieldsizes, out_width, out_depth)
        
        return GeometricTensor(pooled_tensor, FieldType(input.type.gspace, out_height*self.in_fields))
    
    
    def evaluate_output_shape(self, input_shape: tuple) -> tuple:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        
        batch, _, in_width, in_depth = input_shape
        
        out_height = self.in_height // self.v_kernel_size
        out_width = in_width // self.h_kernel_size
        out_depth = in_depth // self.h_kernel_size
        
        return (batch, out_height*self.out_type.size, out_width, out_depth)
    
    

class RBUpsampling(enn.EquivariantModule):
    def __init__(self, gspace: GSpace, in_fields: list[Representation], in_dims: tuple, v_scale: int, h_scale: int):
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
        assert input.type == self.in_type
        
        tensor = input.tensor.reshape(-1, self.in_height, sum(field.size for field in self.in_fields), *self.in_dims[:2])
        tensor = tensor.permute(0, 2, 3, 1, 4)
        
        pooled_tensor = F.interpolate(tensor, scale_factor=[self.h_scale, self.v_scale, self.h_scale], mode='trilinear')
        
        pooled_tensor = pooled_tensor.permute(0, 3, 1, 2, 4)
        batch, out_height, fieldsizes, out_width, out_depth = pooled_tensor.shape
        pooled_tensor = pooled_tensor.reshape(batch, out_height*fieldsizes, out_width, out_depth)
        
        return GeometricTensor(pooled_tensor, FieldType(input.type.gspace, out_height*self.in_fields))
    
    
    def evaluate_output_shape(self, input_shape: tuple) -> tuple:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        
        batch, _, in_width, in_depth = input_shape
        
        out_height = self.in_height * self.v_scale
        out_width = in_width * self.h_scale
        out_depth = in_depth * self.h_scale
        
        return (batch, out_height*self.out_type.size, out_width, out_depth)