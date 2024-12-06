import torch
import numpy as np

from typing import Any
from collections import OrderedDict

from escnn import nn as enn
from escnn.nn import GeometricTensor
from escnn.gspaces import GSpace
from escnn import gspaces

from torch import Tensor

from networks.rayleigh_benard.steerable import RBSteerableConv, RBMaxPool, RBUpsampling

class RBModel(enn.SequentialModule):
    def __init__(self, gspace: GSpace = gspaces.flipRot2dOnR2(N=4),
                   rb_dims: tuple = (48, 48, 32),
                   v_kernel_size: int = 3,
                   h_kernel_size: int = 5,
                   v_pool_size: int = 2,
                   h_pool_size: int = 2,
                   v_upsampling: int = 2,
                   h_upsampling: int = 2,
                   drop_rate: float = 0.2,
                   hidden_channels: tuple = 2*(10,10)
                   ):      
    
        rb_fields = [gspace.trivial_repr, gspace.irrep(1, 1), gspace.trivial_repr]
        hidden_field_type = [gspace.regular_repr]
        
        layers = OrderedDict()
        
        # FIRST BLOCK
        conv = RBSteerableConv(gspace=gspace, 
                               in_fields=rb_fields, 
                               out_fields=hidden_channels[0]*hidden_field_type, 
                               in_dims=rb_dims,
                               v_kernel_size=v_kernel_size, 
                               h_kernel_size=h_kernel_size)
        block = enn.SequentialModule(conv,
                                     enn.ReLU(conv.out_type),
                                     enn.InnerBatchNorm(conv.out_type),
                                     enn.PointwiseDropout(conv.out_type, p=drop_rate))
        layers['Block1'] = block
        self.in_dims = conv.in_dims
        
        
        # POOLING
        pooling = RBMaxPool(gspace=gspace, in_fields=conv.out_fields, in_dims=conv.out_dims, v_kernel_size=v_pool_size, h_kernel_size=h_pool_size)
        layers['Pooling'] = pooling
        
        
        # SECOND BLOCK
        conv = RBSteerableConv(gspace=gspace, 
                               in_fields=pooling.out_fields, 
                               out_fields=hidden_channels[1]*hidden_field_type, 
                               in_dims=pooling.out_dims,
                               v_kernel_size=v_kernel_size, 
                               h_kernel_size=h_kernel_size)
        block = enn.SequentialModule(conv,
                                     enn.ReLU(conv.out_type),
                                     enn.InnerBatchNorm(conv.out_type),
                                     enn.PointwiseDropout(conv.out_type, p=drop_rate))
        layers['Block2'] = block
        
        # UPSAMPLING
        upsampling = RBUpsampling(gspace=gspace, in_fields=conv.out_fields, in_dims=conv.out_dims, v_scale=v_upsampling, h_scale=h_upsampling)
        layers['Upsampling'] = upsampling
        
        
        # OUTPUT LAYER
        conv = RBSteerableConv(gspace=gspace, 
                            in_fields=upsampling.out_fields, 
                            out_fields=rb_fields,
                            in_dims=upsampling.out_dims,
                            v_kernel_size=v_kernel_size, 
                            h_kernel_size=h_kernel_size)
        layers['Block3'] = conv
        self.out_dims = conv.out_dims
        
        
        super().__init__(layers)
        
        
    def forward(self, input: Tensor, data_augmentation: bool = True, on_geometric_tensor: bool = False) -> Tensor:
        if not on_geometric_tensor:
            input = GeometricTensor(input, self.in_type)
        
        if data_augmentation and self.training:
            transformation = self.in_type.gspace.fibergroup.sample()
            input = input.transform(transformation)
            
        out = super().forward(input)
        
        if on_geometric_tensor:
            return out
        else:
            return out.tensor
        
        
    def check_equivariance(self, atol: float = 1e-4, rtol: float = 1e-5) -> list[tuple[Any, float]]:
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
        for el in self.in_type.testing_elements:
            
            out1 = self(x, on_geometric_tensor=True).transform(el).tensor.detach().numpy()
            out2 = self(x.transform(el), on_geometric_tensor=True).tensor.detach().numpy()
        
            errs = out1 - out2
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
        
            assert np.allclose(out1, out2, atol=atol, rtol=rtol), \
                f'The error found during equivariance check with element "{el}" \
                    is too high: max = {errs.max()}, mean = {errs.mean()} var ={errs.var()}'
            
            errors.append((el, errs.mean()))
            
        self.train(training)
        
        return errors