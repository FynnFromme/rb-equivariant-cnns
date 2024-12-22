import torch
import numpy as np

from prettytable import PrettyTable

from typing import Any
from collections import OrderedDict

from escnn import nn as enn
from escnn.nn import GeometricTensor
from escnn.gspaces import GSpace
from escnn import gspaces

from torch import Tensor

from networks.rayleigh_benard.steerable import RBSteerableConv, RBPooling, RBUpsampling

#? info: enn.R2DConv uses he initialization

class _SteerableConvBlock(enn.SequentialModule):
    def __init__(self, 
                 gspace: GSpace,
                 in_fields: list,
                 out_fields: list,
                 in_dims: tuple,
                 v_kernel_size: int,
                 h_kernel_size: int,
                 input_drop_rate: float,
                 bias: bool = True,
                 nonlinearity: bool = True,
                 batch_norm: bool = True):
        
        layers = []
        
        conv = RBSteerableConv(gspace=gspace, 
                               in_fields=in_fields, 
                               out_fields=out_fields, 
                               in_dims=in_dims,
                               v_kernel_size=v_kernel_size, 
                               h_kernel_size=h_kernel_size,
                               bias=bias and not batch_norm, # bias has no effect when using batch norm
                               v_stride=1, h_stride=1,
                               v_pad_mode='zero', h_pad_mode='circular')
        
        if input_drop_rate > 0: layers.append(enn.PointwiseDropout(conv.in_type, p=input_drop_rate))
        layers.append(conv)
        if batch_norm: layers.append(enn.InnerBatchNorm(conv.out_type))
        if nonlinearity: layers.append(enn.ELU(conv.out_type))
        
        super().__init__(*layers)
        
        self.in_dims, self.out_dims = in_dims, conv.out_dims
        self.in_fields, self.out_fields = in_fields, out_fields
        

class RBSteerableAutoEncoder(enn.SequentialModule):
    def __init__(self, 
                 gspace: GSpace,
                 rb_dims: tuple,
                 encoder_channels: tuple,
                 latent_channels: int,
                 v_kernel_size: int = 3,
                 h_kernel_size: int = 3,
                 drop_rate: float = 0.2):
        
        irrep_frequencies = (1, 1) if gspace.flips_order > 0 else (1,) # depending whether using Cn or Dn group
            
        rb_fields = [gspace.trivial_repr, gspace.irrep(*irrep_frequencies), gspace.trivial_repr]
        hidden_field_type = [gspace.regular_repr]
        
        layers = []
        self.out_shapes = OrderedDict()
        self.layer_params = OrderedDict()
        self.out_shapes['Input'] = [sum(f.size for f in rb_fields), 1, *rb_dims]
        
        # Encoder
        in_fields, in_dims = rb_fields, rb_dims
        for i, out_channnels in enumerate(encoder_channels, 1):
            out_fields = out_channnels*hidden_field_type
            layer_drop_rate = 0 if i == 1 else drop_rate
            
            layers.append(_SteerableConvBlock(gspace=gspace, in_fields=in_fields, out_fields=out_fields, 
                                    in_dims=in_dims, v_kernel_size=v_kernel_size, h_kernel_size=h_kernel_size,
                                    input_drop_rate=layer_drop_rate, nonlinearity=True, batch_norm=True))
            in_fields = layers[-1].out_fields
            self.out_shapes[f'EncoderConv{i}'] = [out_channnels, sum(f.size for f in hidden_field_type), *in_dims]
            self.layer_params[f'EncoderConv{i}'] = _count_params(layers[-1])
            
            layers.append(RBPooling(gspace=gspace, in_fields=in_fields, in_dims=in_dims,
                                    v_kernel_size=2, h_kernel_size=2))
            in_dims = layers[-1].out_dims
            self.out_shapes[f'Pooling{i}'] = [out_channnels, sum(f.size for f in hidden_field_type), *in_dims]
            self.layer_params[f'Pooling{i}'] = _count_params(layers[-1])
            
        # Latent Space
        out_fields = latent_channels*hidden_field_type
        layers.append(_SteerableConvBlock(gspace=gspace, in_fields=in_fields, out_fields=out_fields, 
                                          in_dims=in_dims, v_kernel_size=v_kernel_size, h_kernel_size=h_kernel_size,
                                          input_drop_rate=drop_rate, nonlinearity=True, batch_norm=True))
        in_fields = layers[-1].out_fields
        self.out_shapes[f'LatentConv'] = [latent_channels, sum(f.size for f in hidden_field_type), *in_dims]
        self.layer_params[f'LatentConv'] = _count_params(layers[-1])
            
        self.latent_shape = [latent_channels, sum(f.size for f in hidden_field_type), *in_dims]
            
        # Decoder
        for i, out_channnels in enumerate(reversed(encoder_channels), 1):
            out_fields = out_channnels*hidden_field_type
            
            layers.append(_SteerableConvBlock(gspace=gspace, in_fields=in_fields, out_fields=out_fields, 
                                    in_dims=in_dims, v_kernel_size=v_kernel_size, h_kernel_size=h_kernel_size,
                                    input_drop_rate=drop_rate, nonlinearity=True, batch_norm=True))
            in_fields = layers[-1].out_fields
            self.out_shapes[f'DecoderConv{i}'] = [out_channnels, sum(f.size for f in hidden_field_type), *in_dims]
            self.layer_params[f'DecoderConv{i}'] = _count_params(layers[-1])
            
            layers.append(RBUpsampling(gspace=gspace, in_fields=in_fields, in_dims=in_dims,
                                       v_scale=2, h_scale=2))
            in_dims = layers[-1].out_dims
            self.out_shapes[f'Upsampling{i}'] = [out_channnels, sum(f.size for f in hidden_field_type), *in_dims]
            self.layer_params[f'Upsampling{i}'] = _count_params(layers[-1])
        
        # Out Conv
        layers.append(_SteerableConvBlock(gspace=gspace, in_fields=in_fields, out_fields=rb_fields, 
                                 in_dims=in_dims, v_kernel_size=v_kernel_size, h_kernel_size=h_kernel_size,
                                 input_drop_rate=drop_rate, nonlinearity=False, batch_norm=False))
        self.out_shapes['OutputConv'] = [sum(f.size for f in rb_fields), 1, *in_dims]
        self.layer_params['OutputConv'] = _count_params(layers[-1])
        
        self.in_fields, self.out_fields = layers[0].in_fields, layers[-1].out_fields
        self.in_dims, self.out_dims = tuple(layers[0].in_dims), tuple(layers[-1].out_dims)
        
        assert self.out_dims == self.in_dims == tuple(rb_dims)
        assert self.out_fields == self.in_fields == rb_fields
        
        super().__init__(*layers)
        
        
    def forward(self, input: Tensor | GeometricTensor) -> Tensor | GeometricTensor:
        got_geom_tensor = isinstance(input, GeometricTensor)
        if not got_geom_tensor:
            input = GeometricTensor(input, self.in_type)
            
        out = super().forward(input)
        
        if got_geom_tensor:
            return out
        else:
            return out.tensor
            
        
    def check_equivariance(self, atol: float = 1e-4, rtol: float = 1e-5, gpu_device=None) -> list[tuple[Any, float]]:
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
        if gpu_device is not None: 
            x = x.to(gpu_device)
        x = GeometricTensor(x, self.in_type)
        
        errors = []
        for el in self.in_type.testing_elements:
            
            out1 = self(x).transform(el).tensor
            out2 = self(x.transform(el)).tensor
            if gpu_device is not None:
                out1 = out1.cpu()
                out2 = out2.cpu()
            out1 = out1.detach().numpy()
            out2 = out2.detach().numpy()
        
            errs = out1 - out2
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
        
            assert np.allclose(out1, out2, atol=atol, rtol=rtol), \
                f'The error found during equivariance check with element "{el}" \
                    is too high: max = {errs.max()}, mean = {errs.mean()} var ={errs.var()}'
            
            errors.append((el, errs.mean()))
            
        self.train(training)
        
        return errors
    
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