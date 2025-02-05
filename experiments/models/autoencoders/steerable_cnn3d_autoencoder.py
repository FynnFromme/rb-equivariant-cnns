import experiments.models.model_utils as model_utils
from typing import Any, Callable
from collections import OrderedDict

import numpy as np
import torch
from torch import Tensor

from escnn import nn as enn
from escnn.nn import GeometricTensor
from escnn.group import Representation
from escnn.gspaces import GSpace

from layers.conv.steerable_conv3d import RBSteerable3DConv, RBPooling, RBUpsampling, InnerBatchNorm3D


class _Steerable3DConvBlock(enn.SequentialModule):
    def __init__(self, 
                 gspace: GSpace,
                 in_fields: list[Representation],
                 out_fields: list[Representation],
                 in_dims: tuple,
                 kernel_size: int,
                 input_drop_rate: float,
                 bias: bool = True,
                 nonlinearity: Callable = enn.ELU,
                 batch_norm: bool = True):
        """A steerable convolution block (without vertical parameter sharing) with dropout, 
        batch normalization and nonlinearity.

        Args:
            gspace (GSpace): The group of transformations to be equivariant to. For 
                `utils.flipRot2dOnR3.flipRot2dOnR3(N)` the block is equivariant to horizontal flips 
                and rotations. Use `gspaces.rot2dOnR3(N)` for only rotational equivariance.
            in_fields (list[Representation]): The fields of the layer's input. This corresponds to input channels
                in standard convolutions.
            out_fields (list[Representation]): The fields of the layer's output. This corresponds to output 
                channels in standard convolutions.
            in_dims (tuple): The spatial dimensions of the input data.
            kernel_size (int): The horizontal kernel size (in all dimensions).
            input_drop_rate (float): The drop rate for dropout applied to the input of the conv block. Set to 0
                to turn off dropout.
            bias (bool, optional): Whether to apply a bias to the output of the convolution. 
                Bias is turned off automatically when using batch normalization as a bias has no effect when
                using batch normalization. Defaults to True.
            nonlinearity (Callable, optional): The nonlinearity applied to the conv output. Set to `None` to
                have no nonlinearity. Defaults to enn.ELU.
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to True.
        """
        
        conv = RBSteerable3DConv(gspace=gspace, 
                                 in_fields=in_fields, 
                                 out_fields=out_fields, 
                                 in_dims=in_dims,
                                 kernel_size=kernel_size, 
                                 bias=bias and not batch_norm, # bias has no effect when using batch norm
                                 stride=1,
                                 v_pad_mode='zeros', h_pad_mode='circular')
        
        layers = []
        if input_drop_rate > 0: layers.append(enn.PointwiseDropout(conv.in_type, p=input_drop_rate))
        layers.append(conv)
        if batch_norm: layers.append(InnerBatchNorm3D(conv.out_type)) # TODO does that work?
        if nonlinearity: layers.append(nonlinearity(conv.out_type))
        
        super().__init__(*layers)
        
        self.in_dims, self.out_dims = in_dims, conv.out_dims
        self.in_fields, self.out_fields = in_fields, out_fields
        

class RB3DSteerableAutoencoder(enn.EquivariantModule):
    def __init__(self, 
                 gspace: GSpace,
                 rb_dims: tuple,
                 encoder_channels: tuple,
                 latent_channels: int,
                 kernel_size: int,
                 latent_kernel_size: int,
                 drop_rate: float,
                 pool_layers: tuple[bool] = None,
                 nonlinearity: Callable = enn.ELU,
                 **kwargs):
        """A Rayleigh-Bénard autoencoder based on 3D steerable convolutions with vertical parameter sharing.

        Args:
            gspace (GSpace): The group of transformations to be equivariant to. For 
                `utils.flipRot2dOnR3.flipRot2dOnR3(N)` the block is equivariant to horizontal flips 
                and rotations. Use `gspaces.rot2dOnR3(N)` for only rotational equivariance.
            rb_dims (tuple): The spatial dimensions of the simulation data.
            encoder_channels (tuple): The channels of the encoder. Each entry results in a corresponding layer.
                The decoder uses the channels in reversed order.
            latent_channels (int): The number of channels in the latent space.
            kernel_size (int): The kernel size (in all dimensions).
            latent_kernel_size (int): The kernel size (in all dimensions) applied on the latent space.
            drop_rate (float): The drop rate used for dropout. Set to 0 to turn off dropout. 
            pool_layers (tuple[bool], optional): A boolean tuple specifying the encoder layer to pool afterwards.
                The same is used in reversed order for upsampling in the decoder. Defaults to pooling/upsampling
                after each layer.
            nonlinearity (Callable, optional): The nonlinearity applied to the conv output. Set to `None` to
                have no nonlinearity. Defaults to enn.ELU.
        """
        
        super().__init__()
        if pool_layers is None: pool_layers = [True]*len(encoder_channels)
        
        irrep_frequencies = (1, 1) if gspace._sg_id[1] == True else (1,) # depending whether using Cn or Dn group

        rb_fields = [gspace.trivial_repr, gspace.irrep(*irrep_frequencies), gspace.trivial_repr]
        hidden_field_type = [gspace.regular_repr]
        
        encoder_layers = []
        decoder_layers = []
        self.out_shapes = OrderedDict()
        self.layer_params = OrderedDict()
        self.out_shapes['Input'] = [sum(f.size for f in rb_fields), 1, *rb_dims]
        
        #####################
        ####   Encoder   ####
        #####################
        in_fields, in_dims = rb_fields, rb_dims
        for i, (out_channels, pool) in enumerate(zip(encoder_channels, pool_layers), 1):
            out_fields = out_channels*hidden_field_type
            layer_drop_rate = 0 if i == 1 else drop_rate # don't apply dropout to the network's input
            
            encoder_layers.append(_Steerable3DConvBlock(gspace=gspace, 
                                                        in_fields=in_fields, 
                                                        out_fields=out_fields, 
                                                        in_dims=in_dims, 
                                                        kernel_size=kernel_size, 
                                                        input_drop_rate=layer_drop_rate, 
                                                        nonlinearity=nonlinearity, 
                                                        batch_norm=True))
            in_fields = encoder_layers[-1].out_fields
            self.out_shapes[f'EncoderConv{i}'] = [out_channels, sum(f.size for f in hidden_field_type), *in_dims]
            self.layer_params[f'EncoderConv{i}'] = model_utils.count_trainable_params(encoder_layers[-1])
            
            if pool:
                encoder_layers.append(RBPooling(gspace=gspace, 
                                                in_fields=in_fields, 
                                                in_dims=in_dims,
                                                v_kernel_size=2, 
                                                h_kernel_size=2))
                self.out_shapes[f'Pooling{i}'] = [out_channels, sum(f.size for f in hidden_field_type), 
                                                  *encoder_layers[-1].out_dims]
                self.layer_params[f'Pooling{i}'] = model_utils.count_trainable_params(encoder_layers[-1])
            in_dims = encoder_layers[-1].out_dims
            
        ######################
        #### Latent Space ####
        ######################
        out_fields = latent_channels*hidden_field_type
        encoder_layers.append(_Steerable3DConvBlock(gspace=gspace, 
                                                    in_fields=in_fields, 
                                                    out_fields=out_fields, 
                                                    in_dims=in_dims, 
                                                    kernel_size=latent_kernel_size, 
                                                    input_drop_rate=drop_rate, 
                                                    nonlinearity=nonlinearity, 
                                                    batch_norm=True))
        in_fields = encoder_layers[-1].out_fields
        self.out_shapes[f'LatentConv'] = [latent_channels, sum(f.size for f in hidden_field_type), *in_dims]
        self.layer_params[f'LatentConv'] = model_utils.count_trainable_params(encoder_layers[-1])
            
        self.latent_shape = [latent_channels, sum(f.size for f in hidden_field_type), *in_dims]
            
        #####################
        ####   Decoder   ####
        #####################
        decoder_channels = reversed(encoder_channels)
        upsample_layers = reversed(pool_layers)
        for i, (out_channels, upsample) in enumerate(zip(decoder_channels, upsample_layers), 1):
            out_fields = out_channels*hidden_field_type
            
            decoder_layers.append(_Steerable3DConvBlock(gspace=gspace, 
                                                        in_fields=in_fields, 
                                                        out_fields=out_fields, 
                                                        in_dims=in_dims, 
                                                        kernel_size=kernel_size,
                                                        input_drop_rate=drop_rate,
                                                        nonlinearity=nonlinearity, 
                                                        batch_norm=True))
            in_fields = decoder_layers[-1].out_fields
            self.out_shapes[f'DecoderConv{i}'] = [out_channels, sum(f.size for f in hidden_field_type), *in_dims]
            self.layer_params[f'DecoderConv{i}'] = model_utils.count_trainable_params(decoder_layers[-1])
            
            if upsample:
                decoder_layers.append(RBUpsampling(gspace=gspace, 
                                                in_fields=in_fields, 
                                                in_dims=in_dims,
                                                v_scale=2, 
                                                h_scale=2))
                self.out_shapes[f'Upsampling{i}'] = [out_channels, sum(f.size for f in hidden_field_type), 
                                                     *decoder_layers[-1].out_dims]
                self.layer_params[f'Upsampling{i}'] = model_utils.count_trainable_params(decoder_layers[-1])
            in_dims = decoder_layers[-1].out_dims
        
        ######################
        ####    Output    ####
        ######################
        decoder_layers.append(_Steerable3DConvBlock(gspace=gspace, 
                                                    in_fields=in_fields, 
                                                    out_fields=rb_fields, 
                                                    in_dims=in_dims, 
                                                    kernel_size=kernel_size, 
                                                    input_drop_rate=drop_rate, 
                                                    nonlinearity=None, 
                                                    batch_norm=False))
        self.out_shapes['OutputConv'] = [sum(f.size for f in rb_fields), 1, *in_dims]
        self.layer_params['OutputConv'] = model_utils.count_trainable_params(decoder_layers[-1])
        
        self.in_type, self.out_type = encoder_layers[0].in_type, decoder_layers[-1].out_type
        self.in_fields, self.out_fields = encoder_layers[0].in_fields, decoder_layers[-1].out_fields
        self.in_dims, self.out_dims = tuple(encoder_layers[0].in_dims), tuple(decoder_layers[-1].out_dims)
        
        assert self.out_dims == self.in_dims == tuple(rb_dims)
        assert self.out_fields == self.in_fields == rb_fields
        
        self.encoder = enn.SequentialModule(*encoder_layers)
        self.decoder = enn.SequentialModule(*decoder_layers)
        
    
    def train(self, *args, **kwargs):
        """Sets module to training mode."""
        self.encoder.train(*args, **kwargs)
        self.decoder.train(*args, **kwargs)
    
    
    def eval(self, *args, **kwargs):
        """Sets module to evaluation mode."""
        self.encoder.eval(*args, **kwargs)
        self.decoder.eval(*args, **kwargs)
    
        
    def forward(self, input: Tensor) -> Tensor:
        """Forwards the input through the network and returns the output.

        Args:
            input (Tensor): The networks input of shape [batch, width, depth, height, channels]

        Returns:
            Tensor: The decoded output of shape [batch, width, depth, height, channels]
        """
        input = self._from_input_shape(input)
        
        input = GeometricTensor(input, self.encoder.in_type)
        latent = self.encoder(input)
        output = self.decoder(latent)
        output = output.tensor
        
        return self._to_output_shape(output)
    
    
    def forward_geometric(self, input: GeometricTensor) -> GeometricTensor:
        """Forwards the input through the network and returns the output. This method works with
        the GeometricTensor, which is internally used.

        Args:
            input (GeometricTensor): The networks input of shape [batch, sum(fieldsizes), height, width, depth]

        Returns:
            GeometricTensor: The decoded output of shape [batch, sum(fieldsizes), height, width, depth]
        """
        latent = self.encoder(input)
        output = self.decoder(latent)
        
        return output
    
    
    def encode(self, input: Tensor) -> Tensor:
        """Forwards the input through the encoder part and returns the latent representation.

        Args:
            input (Tensor): The networks input of shape [batch, width, depth, height, channels]

        Returns:
            Tensor: The latent representation of shape [batch, width, depth, height, sum(fieldsizes)]
        """
        input = self._from_input_shape(input)
        
        input = GeometricTensor(input, self.encoder.in_type)
        latent = self.encoder(input)
        latent = latent.tensor
        
        return self._to_latent_shape(latent)
    
    
    def decode(self, latent: Tensor) -> Tensor:
        """Forwards the latent representation through the decoder part and returns the decoded output.

        Args:
            input (Tensor): The latent representation of shape [batch, width, depth, height, sum(fieldsizes)]

        Returns:
            Tensor: The decoded output of shape [batch, width, depth, height, channels]
        """
        latent = self._from_latent_shape(latent)
        
        latent = GeometricTensor(latent, self.decoder.in_type)
        output = self.decoder(latent)
        output = output.tensor
        
        return self._to_output_shape(output)
    
    
    def _from_input_shape(self, tensor: Tensor) -> Tensor:
        """Transforms an input tensor of shape [batch, width, depth, height, sum(fieldsizes)] into the
        shape required for this model.

        Args:
            tensor (Tensor): Tensor of shape [batch, width, depth, height, sum(fieldsizes)].

        Returns:
            Tensor: Transformed tensor of shape [batch, sum(fieldsizes), height, width, depth]
        """
        return tensor.permute(0, 4, 3, 1, 2)
    
    
    def _to_output_shape(self, tensor: Tensor) -> Tensor:
        """Transforms the output of the model into the desired shape of the output:
        [batch, width, depth, height, sum(fieldsizes)]

        Args:
            tensor (Tensor): Tensor of shape [batch, sum(fieldsizes), height, width, depth]

        Returns:
            Tensor: Transformed tensor of shape [batch, width, depth, height, sum(fieldsizes)]
        """
        return tensor.permute(0, 3, 4, 2, 1)
    
    
    def _to_latent_shape(self, tensor: Tensor) -> Tensor:
        """Transforms the output of the encoder model into the desired 
        shape of the latent representation: [batch, width, depth, height, sum(fieldsizes)]

        Args:
            tensor (Tensor): Tensor of shape [batch, sum(fieldsizes), height, width, depth]

        Returns:
            Tensor: Transformed tensor of shape [batch, width, depth, height, sum(fieldsizes)]
        """
        return tensor.permute(0, 3, 4, 2, 1)
    
    
    def _from_latent_shape(self, tensor: Tensor) -> Tensor:
        """Transforms an latent representation of shape [batch, width, depth, height, sum(fieldsizes)] 
        into the shape required for the decoder model

        Args:
            tensor (Tensor): Tensor of shape [batch, width, depth, height, sum(fieldsizes)].

        Returns:
            Tensor: Transformed tensor of shape [batch, sum(fieldsizes), height, width, depth]
        """
        return tensor.permute(0, 4, 3, 1, 2)
    
    
    def evaluate_output_shape(self, input_shape: tuple) -> tuple:
        """Compute the shape the output tensor which would be generated by this module when a tensor with shape
        ``input_shape`` is provided as input.
        
        Args:
            input_shape (tuple): shape of the input tensor

        Returns:
            shape of the output tensor
            
        """
        return input_shape
            
        
    def check_equivariance(self, atol: float = 1e-4, rtol: float = 1e-5, gpu_device=None) -> list[tuple[Any, float]]:
        """Method that automatically tests the equivariance of the current module.
        
        Returns:
            list: A list containing containing for each testing element a pair with that element and the 
            corresponding equivariance error
        """
        
        training = self.training
        self.eval()
        
        x = torch.randn(3, self.in_dims[2], *self.in_dims[:2], 4)
        x = self._from_input_shape(x)
        if gpu_device is not None: 
            x = x.to(gpu_device)
        x = GeometricTensor(x, self.in_type)
        
        errors = []
        for el in self.in_type.testing_elements:
            out1 = self.forward_geometric(x).transform(el).tensor
            out2 = self.forward_geometric(x.transform(el)).tensor
            
            out1 = self._to_output_shape(out1)
            out2 = self._to_output_shape(out2)
            
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
        """Print summary of the model."""
        model_utils.summary(self, self.out_shapes, self.layer_params, self.latent_shape)