import experiments.models.model_utils as model_utils
from typing import Any, Callable, Literal
from collections import OrderedDict

import numpy as np
import torch
from torch import Tensor

from escnn import nn as enn
from escnn.nn import GeometricTensor
from escnn.group import Representation
from escnn.gspaces import GSpace

from layers.conv.steerable_conv import RBSteerableConv, RBPooling, RBUpsampling


class _SteerableConvBlock(enn.SequentialModule):
    def __init__(self, 
                 gspace: GSpace,
                 in_fields: list[Representation],
                 out_fields: list[Representation],
                 in_dims: tuple,
                 v_kernel_size: int,
                 h_kernel_size: int,
                 v_share : int,
                 input_drop_rate: float,
                 bias: bool = True,
                 nonlinearity: Callable = enn.ELU,
                 batch_norm: bool = True):
        """A steerable convolution block (without vertical parameter sharing) with dropout, 
        batch normalization and nonlinearity.

        Args:
            gspace (GSpace): The group of transformations to be equivariant to. For `gspaces.flipRot2dOnR2(N)`
                the block is equivariant to horizontal flips and rotations. Use `gspaces.rot2dOnR2(N)` for only
                rotational equivariance.
            in_fields (list[Representation]): The fields of the layer's input. This corresponds to input channels
                in standard convolutions.
            out_fields (list[Representation]): The fields of the layer's output. This corresponds to output 
                channels in standard convolutions.
            in_dims (tuple): The spatial dimensions of the input data.
            v_kernel_size (int): The vertical kernel size.
            h_kernel_size (int): The horizontal kernel size (in both directions).
            v_share (int): The number of neighboring output-heights sharing the same kernel.
            input_drop_rate (float): The drop rate for dropout applied to the input of the conv block. Set to 0
                to turn off dropout.
            bias (bool, optional): Whether to apply a bias to the output of the convolution. 
                Bias is turned off automatically when using batch normalization as a bias has no effect when
                using batch normalization. Defaults to True.
            nonlinearity (Callable, optional): The nonlinearity applied to the conv output. Set to `None` to
                have no nonlinearity. Defaults to enn.ELU.
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to True.
        """
        
        conv = RBSteerableConv(gspace=gspace, 
                               in_fields=in_fields, 
                               out_fields=out_fields, 
                               in_dims=in_dims,
                               v_kernel_size=v_kernel_size, 
                               h_kernel_size=h_kernel_size,
                               v_share=v_share,
                               bias=bias and not batch_norm, # bias has no effect when using batch norm
                               v_stride=1, h_stride=1,
                               v_pad_mode='zero', h_pad_mode='circular')
        
        layers = []
        if input_drop_rate > 0: layers.append(enn.PointwiseDropout(conv.in_type, p=input_drop_rate))
        layers.append(conv)
        if batch_norm: layers.append(enn.InnerBatchNorm(conv.out_type))
        if nonlinearity: layers.append(nonlinearity(conv.out_type))
        
        super().__init__(*layers)
        
        self.in_dims, self.out_dims = in_dims, conv.out_dims
        self.in_fields, self.out_fields = in_fields, out_fields
        

class RBSteerableAutoencoder(enn.EquivariantModule):
    def __init__(self, 
                 gspace: GSpace,
                 rb_dims: tuple,
                 encoder_channels: tuple,
                 latent_channels: int,
                 v_kernel_size: int,
                 h_kernel_size: int,
                 latent_v_kernel_size: int,
                 latent_h_kernel_size: int,
                 drop_rate: float,
                 v_shares: tuple = None,
                 pool_layers: tuple[bool] = None,
                 nonlinearity: Callable = enn.ELU,
                 **kwargs):
        """A Rayleigh-Bénard autoencoder based on steerable convolutions without vertical parameter sharing.

        Args:
            gspace (GSpace): The group of transformations to be equivariant to. For `gspaces.flipRot2dOnR2(N)`
                the block is equivariant to horizontal flips and rotations. Use `gspaces.rot2dOnR2(N)` for only
                rotational equivariance.
            rb_dims (tuple): The spatial dimensions of the simulation data.
            encoder_channels (tuple): The channels of the encoder. Each entry results in a corresponding layer.
                The decoder uses the channels in reversed order.
            latent_channels (int): The number of channels in the latent space.
            v_kernel_size (int): The vertical kernel size.
            h_kernel_size (int): The horizontal kernel size (in both directions).
            v_shares (tuple): The number of neighboring output-heights sharing the same kernel for each encoder 
                layer (and latent convolution!). Therefore, len(v_shares)=len(encoder_channels)+1. The same is 
                used in reversed order in the decoder. Defaults to a 1-tuple.
            latent_v_kernel_size (int): The vertical kernel size applied on the latent space.
            latent_h_kernel_size (int): The horizontal kernel size (in both directions) applied on the latent space.
            drop_rate (float): The drop rate used for dropout. Set to 0 to turn off dropout.
            pool_layers (tuple[bool], optional): A boolean tuple specifying the encoder layer to pool afterwards.
                The same is used in reversed order for upsampling in the decoder. Defaults to pooling/upsampling
                after each layer.
            nonlinearity (Callable, optional): The nonlinearity applied to the conv output. Set to `None` to
                have no nonlinearity. Defaults to enn.ELU.
        """
        super().__init__()
        if v_shares is not None:
            assert len(v_shares) == len(encoder_channels)+1, 'required to specify v_share also for latent conv'
        else:
            v_shares = [1]*(len(encoder_channels)+1)
        
        if pool_layers is None: pool_layers = [True]*len(encoder_channels)
        
        irrep_frequencies = (1, 1) if gspace.flips_order > 0 else (1,) # depending whether using Dn or Cn group
            
        rb_fields = [gspace.trivial_repr, gspace.irrep(*irrep_frequencies), gspace.trivial_repr]
        self.hidden_field_type = [gspace.regular_repr]
        
        self.encoder_layers = OrderedDict()
        self.decoder_layers = OrderedDict()
        
        #####################
        ####   Encoder   ####
        #####################
        fields, dims = rb_fields, rb_dims
        for i, (out_channels, v_share, pool) in enumerate(zip(encoder_channels, v_shares, pool_layers), 1):
            conv = _SteerableConvBlock(gspace=gspace, 
                                       in_fields=fields, 
                                       out_fields=out_channels*self.hidden_field_type, 
                                       in_dims=dims, 
                                       v_kernel_size=v_kernel_size, 
                                       h_kernel_size=h_kernel_size,
                                       v_share=v_share,
                                       input_drop_rate=0 if i == 1 else drop_rate, # don't apply dropout to the networks input
                                       nonlinearity=nonlinearity, 
                                       batch_norm=True)
            self.encoder_layers[f'EncoderConv{i}'] = conv
            fields = conv.out_fields
            
            if pool:
                pool = RBPooling(gspace=gspace, in_fields=fields, in_dims=dims, v_kernel_size=2, h_kernel_size=2)
                self.encoder_layers[f'Pooling{i}'] = pool
                dims = pool.out_dims
            
        ######################
        #### Latent Space ####
        ######################
        conv = _SteerableConvBlock(gspace=gspace, 
                                   in_fields=fields, 
                                   out_fields=latent_channels*self.hidden_field_type, 
                                   in_dims=dims, 
                                   v_kernel_size=latent_v_kernel_size, 
                                   h_kernel_size=latent_h_kernel_size,
                                   v_share=v_shares[-1],
                                   input_drop_rate=drop_rate, 
                                   nonlinearity=nonlinearity, 
                                   batch_norm=True)
        self.encoder_layers[f'LatentConv'] = conv
        fields = conv.out_fields
            
        self.latent_shape = [*dims, sum(f.size for f in latent_channels*self.hidden_field_type)]
            
        #####################
        ####   Decoder   ####
        #####################
        decoder_channels = reversed(encoder_channels)
        upsample_layers = reversed(pool_layers)
        decoder_v_shares = list(reversed(v_shares))
        for i, (out_channels, v_share, upsample) in enumerate(zip(decoder_channels, decoder_v_shares, upsample_layers), 1):            
            conv = _SteerableConvBlock(gspace=gspace, 
                                       in_fields=fields, 
                                       out_fields=out_channels*self.hidden_field_type, 
                                       in_dims=dims, 
                                       v_kernel_size=v_kernel_size, 
                                       h_kernel_size=h_kernel_size,
                                       v_share=v_share,
                                       input_drop_rate=drop_rate,
                                       nonlinearity=nonlinearity, 
                                       batch_norm=True)
            self.decoder_layers[f'DecoderConv{i}'] = conv
            fields = conv.out_fields
            
            if upsample:
                upsample = RBUpsampling(gspace=gspace, in_fields=fields, in_dims=dims, v_scale=2, h_scale=2)
                self.decoder_layers[f'Upsampling{i}'] = upsample
                dims = upsample.out_dims
        
        ######################
        ####    Output    ####
        ######################
        conv = _SteerableConvBlock(gspace=gspace, 
                                   in_fields=fields, 
                                   out_fields=rb_fields, 
                                   in_dims=dims, 
                                   v_kernel_size=v_kernel_size, 
                                   h_kernel_size=h_kernel_size,
                                   v_share=decoder_v_shares[-1],
                                   input_drop_rate=drop_rate, 
                                   nonlinearity=None, 
                                   batch_norm=False)
        self.decoder_layers['OutputConv'] = conv
        
        first_layer = self.encoder_layers[next(iter(self.encoder_layers))]
        last_layer = self.decoder_layers[next(reversed(self.decoder_layers))]
        self.in_type, self.out_type = first_layer.in_type, last_layer.out_type
        self.in_fields, self.out_fields = first_layer.in_fields, last_layer.out_fields
        self.in_dims, self.out_dims = tuple(first_layer.in_dims), tuple(last_layer.out_dims)
        
        assert self.out_dims == self.in_dims == tuple(rb_dims)
        assert self.out_fields == self.in_fields == rb_fields
        
        self.encoder = enn.SequentialModule(self.encoder_layers)
        self.decoder = enn.SequentialModule(self.decoder_layers)
    
        
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
            input (GeometricTensor): The networks input of shape [batch, sum(fieldsizes)*channels, width, depth]

        Returns:
            GeometricTensor: The decoded output of shape [batch, sum(fieldsizes)*channels, width, depth]
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
            Tensor: Transformed tensor of shape [batch, height*sum(fieldsizes), width, depth]
        """
        b, w, d, h, c = tensor.shape
        return tensor.permute(0, 3, 4, 1, 2).reshape(b, h*c, w, d)
    
    
    def _to_output_shape(self, tensor: Tensor) -> Tensor:
        """Transforms the output of the model into the desired shape of the output:
        [batch, width, depth, height, sum(fieldsizes)]

        Args:
            tensor (Tensor): Tensor of shape [batch, height*sum(fieldsizes), width, depth]

        Returns:
            Tensor: Transformed tensor of shape [batch, width, depth, height, sum(fieldsizes)]
        """
        b = tensor.shape[0]
        w, d, h = self.out_dims
        return tensor.reshape(b, h, 4, w, d).permute(0, 3, 4, 1, 2)
    
    
    def _to_latent_shape(self, tensor: Tensor) -> Tensor:
        """Transforms the output of the encoder model into the desired 
        shape of the latent representation: [batch, width, depth, height, sum(fieldsizes)]

        Args:
            tensor (Tensor): Tensor of shape [batch, height*sum(fieldsizes), width, depth]

        Returns:
            Tensor: Transformed tensor of shape [batch, width, depth, height, sum(fieldsizes)]
        """
        b = tensor.shape[0]
        w, d, h, f = self.latent_shape
        return tensor.reshape(b, h, f, w, d).permute(0, 3, 4, 1, 2)
    
    
    def _from_latent_shape(self, tensor: Tensor) -> Tensor:
        """Transforms an latent representation of shape [batch, width, depth, height, sum(fieldsizes)] 
        into the shape required for the decoder model

        Args:
            tensor (Tensor): Tensor of shape [batch, width, depth, height, sum(fieldsizes)].

        Returns:
            Tensor: Transformed tensor of shape [batch, height*sum(fieldsizes), width, depth]
        """
        b, w, d, h, f = tensor.shape
        return tensor.permute(0, 3, 4, 1, 2).reshape(b, h*f, w, d)
    
    
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
        
        x = torch.randn(3, *self.in_dims, 4)
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
        
    
    def layer_out_shapes(self, part: Literal['encoder', 'decoder', 'both'] = 'both') -> OrderedDict:
        """Computes the output shape for the layers of the autoencoder.

        Args:
            part (Literal['encoder', 'decoder', 'both'], optional): Whether to include all layers or only layers of
                the encoder or decoder. Defaults to 'both'.

        Returns:
            OrderedDict: The output shapes for the layers of the autoencoder.
        """
        layers = OrderedDict()
        if part in ('encoder', 'both'):
            layers.update(self.encoder_layers)
        if part in ('decoder', 'both'):
            layers.update(self.decoder_layers)
        
        out_shapes = OrderedDict()
        out_shapes['Input'] = [*self.in_dims, 4, 1]
        for i, (name, layer) in enumerate(layers.items(), 1):
            if i < len(layers):
                out_shapes[name] = [*layer.out_dims, len(layer.out_fields), sum(f.size for f in self.hidden_field_type)]
            else:
                out_shapes[name] = [*layer.out_dims, 4, 1]
            
        return out_shapes
    
    
    def layer_params(self, part: Literal['encoder', 'decoder', 'both'] = 'both') -> OrderedDict:
        """Computes the number of parameters for the layers of the autoencoder.

        Args:
            part (Literal['encoder', 'decoder', 'both'], optional): Whether to include all layers or only layers of
                the encoder or decoder. Defaults to 'both'.

        Returns:
            OrderedDict: The number of parameters for the layers of the autoencoder.
        """
        layers = OrderedDict()
        if part in ('encoder', 'both'):
            layers.update(self.encoder_layers)
        if part in ('decoder', 'both'):
            layers.update(self.decoder_layers)
        
        params = OrderedDict()
        for i, (name, layer) in enumerate(layers.items(), 1):
            params[name] = model_utils.count_trainable_params(layer)
            
        return params
    
    
    def summary(self):
        """Print summary of the model."""        
        out_shapes = self.layer_out_shapes()
        params = self.layer_params()
            
        model_utils.summary(self, out_shapes, params, steerable=True)
        
        print(f'\nShape of latent space: {out_shapes["LatentConv"]}')
    
        print(f'\nLatent-Input-Ratio: {np.prod(self.latent_shape)/np.prod(out_shapes["Input"])*100:.2f}%')