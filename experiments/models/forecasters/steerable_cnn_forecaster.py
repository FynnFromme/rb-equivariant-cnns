import numpy as np
import torch
import random

from escnn import nn as enn
from escnn.nn import GeometricTensor
from escnn.gspaces import GSpace

from layers.conv.steerable_conv import RBSteerableConv
from layers.lstm.steerable_conv_lstm import RBSteerableConvLSTM

from experiments.models import model_utils
from collections import OrderedDict
from typing import Literal, Any

#! TODO LayerNorm?
class RBSteerableForecaster(enn.EquivariantModule):
    def __init__(
        self,
        gspace: GSpace,
        autoencoder: torch.nn.Module,
        num_layers: int,
        latent_channels: int,
        hidden_channels: list[int],
        latent_dims: tuple[int],
        v_kernel_size: int,
        h_kernel_size: int,
        use_lstm_encoder: bool = True,
        residual_connection: bool = True,
        nonlinearity: Literal['relu', 'elu', 'tanh'] = 'tanh',
        drop_rate: float = 0,
        recurrent_drop_rate: float = 0,
        parallel_ops: bool = True, # applies autoencoder and output layer in parallel (might result in out of memory for large sequences)
        train_autoencoder: bool = False,
        min_forced_decoding_prob: float = 0,
        init_forced_decoding_prob: float = 1,
        forced_decoding_epochs: float = 100,
        backprop_through_autoregression: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.use_lstm_encoder = use_lstm_encoder
        
        self.gspace = gspace
        
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.latent_dims = latent_dims
        self.residual_connection = residual_connection
        
        self.backprop_through_autoregression = backprop_through_autoregression
        
        self.autoencoder = autoencoder
        self.parallel_ops = parallel_ops
        self.train_autoencoder = train_autoencoder
        
        self.min_forced_decoding_prob = min_forced_decoding_prob
        self.init_forced_decoding_prob = init_forced_decoding_prob
        self.forced_decoding_epochs = forced_decoding_epochs
        
        self.field_type = [gspace.regular_repr]
        
        if use_lstm_encoder:
            self.lstm_encoder = RBSteerableConvLSTM(gspace=gspace,
                                                    num_layers=num_layers, 
                                                    in_fields=latent_channels*self.field_type, 
                                                    hidden_fields=[hc*self.field_type for hc in hidden_channels],
                                                    dims=latent_dims, 
                                                    v_kernel_size=v_kernel_size, 
                                                    h_kernel_size=h_kernel_size, 
                                                    nonlinearity=nonlinearity,
                                                    drop_rate=drop_rate,
                                                    recurrent_drop_rate=recurrent_drop_rate,
                                                    bias=True)
        else:
            self.lstm_encoder = None
            
        self.lstm_decoder = RBSteerableConvLSTM(gspace=gspace,
                                                num_layers=num_layers, 
                                                in_fields=latent_channels*self.field_type, 
                                                hidden_fields=[hc*self.field_type for hc in hidden_channels],
                                                dims=latent_dims, 
                                                v_kernel_size=v_kernel_size, 
                                                h_kernel_size=h_kernel_size, 
                                                nonlinearity=nonlinearity,
                                                drop_rate=drop_rate,
                                                recurrent_drop_rate=recurrent_drop_rate,
                                                bias=True)
        
        self.dropout = enn.PointwiseDropout(self.lstm_decoder.out_type, drop_rate)
        
        self.output_conv = RBSteerableConv(gspace=gspace,
                                           in_fields=hidden_channels[-1]*self.field_type,
                                           out_fields=latent_channels*self.field_type,
                                           in_dims=latent_dims,
                                           v_kernel_size=v_kernel_size,
                                           h_kernel_size=h_kernel_size,
                                           v_pad_mode='zero',
                                           h_pad_mode='circular',
                                           bias=True)
        
        first_lstm = self.lstm_encoder if use_lstm_encoder else self.lstm_decoder
        if autoencoder is not None:
            self.in_type, self.out_type = self.autoencoder.in_type, self.autoencoder.out_type
            self.in_fields, self.out_fields = self.autoencoder.in_fields, self.autoencoder.out_fields
            self.in_dims, self.out_dims = self.autoencoder.in_dims, self.autoencoder.out_dims
            self.in_height, self.out_height = self.autoencoder.in_dims[-1], self.autoencoder.out_dims[-1]
        else:
            self.in_type, self.out_type = first_lstm.in_type, self.output_conv.out_type
            self.in_fields, self.out_fields = first_lstm.in_fields, self.output_conv.out_fields
            self.in_dims, self.out_dims = first_lstm.in_dims, self.output_conv.out_dims
            self.in_height, self.out_height = first_lstm.in_height, self.output_conv.out_height
        self.in_latent_type, self.out_latent_type = first_lstm.in_type, self.output_conv.out_type
        self.in_latent_fields, self.out_latent_fields = first_lstm.in_fields, self.output_conv.out_fields
        self.in_latent_dims, self.out_latent_dims = first_lstm.in_dims, self.output_conv.out_dims
        self.in_latent_height, self.out_latent_height = first_lstm.in_height, self.output_conv.out_height
            
        
    def forward(self, warmup_input, steps=1, ground_truth=None, epoch=-1):
        # input shape [batch, seq, width, depth, height, channels]
        
        # does not use the forward_gen generator in order to be able to apply the decoder in parallel to
        # the whole sequence
        
        assert warmup_input.ndim==6, "warmup_input must be a sequence"

        if self.autoencoder is not None:
            # encode into latent space
            warmup_latent = self._encode(warmup_input)
            
            if not self.train_autoencoder:
                warmup_latent = warmup_latent.detach()
        else:
            warmup_latent = warmup_input
        
        output_latent = self.forward_latent(warmup_latent, steps, ground_truth, epoch)

        if self.autoencoder is not None:
            # decode into original space
            output = self._decode(output_latent)
        else:
            output = output_latent
        
        return output
    
    def forward_gen(self, warmup_input, steps=1, ground_truth=None, epoch=-1):
        # input shape [batch, seq, width, depth, height, channels]
        
        assert warmup_input.ndim==6, "warmup_input must be a sequence"

        if self.autoencoder is not None:
            # encode into latent space
            warmup_latent = self._encode(warmup_input)
            
            if not self.train_autoencoder:
                warmup_latent = warmup_latent.detach()
        else:
            warmup_latent = warmup_input
        
        
        for output_latent in self.forward_latent_gen(warmup_latent, steps, ground_truth, epoch):
            if self.autoencoder is not None:
                # decode into original space
                output = self._decode(output_latent)
            else:
                output = output_latent
            
            yield output
    
    def forward_latent(self, warmup_input: torch.Tensor, steps=1, ground_truth=None, epoch=-1):
        outputs = list(self.forward_latent_gen(warmup_input, steps, ground_truth, epoch))
        return torch.stack(outputs, dim=1)
    
    def forward_latent_gen(self, warmup_input: torch.Tensor, steps=1, ground_truth=None, epoch=-1):
        # input shape (b,warmup_seq,c,w,d,h) -> (b,forecast_seq,c,w,d,h) or (b,warmup_preds+forecast_seq,c,w,d,h)
        # 2 phases: warmup (lstm gets ground truth as input), autoregression: (lstm gets its own outputs as input, for steps > 1)
        
        dims = warmup_input.shape[2:5]
        assert tuple(dims) == tuple(self.latent_dims)
        
        warmup_input = self._from_input_shape(warmup_input)
        warmup_length = warmup_input.size(1)
        geom_warmup_input = [GeometricTensor(warmup_input[:, t], self.lstm_decoder.in_type) for t in range(warmup_length)]
        
        if ground_truth is not None:
            ground_truth = self._from_input_shape(ground_truth)
            ground_truth_lenth = ground_truth.size(1)
            assert ground_truth_lenth == steps
            geom_ground_truth = [GeometricTensor(ground_truth[:, t], self.lstm_decoder.in_type) for t in range(steps)]
        else:
            geom_ground_truth = None
        
        for geom_output in self._forward_geometric_latent_gen(geom_warmup_input, steps, geom_ground_truth, epoch):
            yield self._to_output_shape(geom_output.tensor)
    
    def _forward_geometric_latent_gen(self, warmup_input: list[GeometricTensor], steps=1, ground_truth=None, epoch=-1):
        if self.use_lstm_encoder:
            _, encoded_state = self.lstm_encoder.forward(warmup_input)
            decoder_input = [warmup_input[-1]]
        else:
            encoded_state = None
            decoder_input = warmup_input
            
        lstm_autoregressor = self.lstm_decoder.autoregress(decoder_input, steps, encoded_state)
        
        lstm_input = None # warmup_input is already provided to LSTM
        for i in range(steps):
            lstm_out = lstm_autoregressor.send(lstm_input)
            out = self._apply_output_layer(lstm_out)
            
            if self.residual_connection:
                if i == 0:
                    res_input = [warmup_input[-1]]
                else:
                    res_input = lstm_input
                out = [res + o for res, o in zip(res_input, out)]
                
            for geom_output in out:
                yield geom_output
            
            if self.training and ground_truth is not None and epoch >= 0:
                forced_decoding_prob = max(self.min_forced_decoding_prob, 
                                           self.init_forced_decoding_prob * (1 - (epoch-1)/self.forced_decoding_epochs))
            else:
                forced_decoding_prob = 0
            forced_decoding = random.random() < forced_decoding_prob

            if forced_decoding:
                lstm_input = [ground_truth[i]]
            else:
                lstm_input = [out[-1]]
                if not self.backprop_through_autoregression:
                    for inp in lstm_input:
                        inp.tensor = inp.tensor.detach()
            
    
    def _apply_output_layer(self, lstm_out: list[GeometricTensor]):  
        # shape: (b,h*c,w,d,seq)
        
        seq_length = len(lstm_out)
        batch_size, _, w, d = lstm_out[0].shape
        
        if self.parallel_ops:
            # apply output layer in parallel to whole sequence
            lstm_out_flat = self._merge_batch_and_seq_dim(lstm_out)
            
            # lstm_out_flat = self.dropout(lstm_out_flat)
            output_flat = self.output_conv(lstm_out_flat)
            
            outputs = self._split_batch_and_seq_dim(output_flat, batch_size)
        else:
            outputs = []
            for i in range(seq_length):
                hidden_state = lstm_out[i]
                hidden_state = self.dropout(hidden_state)
                outputs.append(self.output_conv(hidden_state))
                   
        return outputs
    
    def _merge_batch_and_seq_dim(self, geom_tensors: list[GeometricTensor]) -> GeometricTensor:
        tensor = torch.cat([geom_tensor.tensor for geom_tensor in geom_tensors], dim=0)
        return GeometricTensor(tensor, geom_tensors[0].type)
    
    def _split_batch_and_seq_dim(self, geom_tensor: GeometricTensor, batch_size: int) -> list[GeometricTensor]:
        tensors = geom_tensor.tensor.split(batch_size, dim=0)
        return [GeometricTensor(tensor, geom_tensor.type) for tensor in tensors]
    
    def _encode(self, input):
        batch_size, seq_length = input.shape[:2]
        
        if self.parallel_ops:
            # apply encoder in parallel to whole sequence
            input_flat = input.reshape(batch_size*seq_length, *input.shape[2:])
            latent_flat = self.autoencoder.encode(input_flat)
            latent = latent_flat.reshape(batch_size, seq_length, *latent_flat.shape[1:])
        else:
            latents = []
            for i in range(seq_length):
                latents.append(self.autoencoder.encode(input[:, i]))
            latent = torch.stack(latents, dim=1)
        
        return latent
    
    def _decode(self, latent: torch.Tensor):
        is_sequence = latent.ndim == 6
        if not is_sequence:
            latent = latent.unsqueeze(1)
        batch_size, seq_length = latent.shape[:2]
        
        if self.parallel_ops:
            # apply encoder in parallel to whole sequence
            latent_flat = latent.reshape(batch_size*seq_length, *latent.shape[2:])
            output_flat = self.autoencoder.decode(latent_flat)
            output = output_flat.reshape(batch_size, seq_length, *output_flat.shape[1:])
        else:
            outputs = []
            for i in range(seq_length):
                outputs.append(self.autoencoder.decode(latent[:, i]))
            output = torch.stack(outputs, dim=1)
        
        if not is_sequence:
            output = output.squeeze(1)
        
        return output
    
    def _from_input_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transforms an input tensor of shape [batch, seq, width, depth, height, sum(fieldsizes)] into the
        shape required for this model.

        Args:
            tensor (Tensor): Tensor of shape [batch, seq, width, depth, height, sum(fieldsizes)].

        Returns:
            Tensor: Transformed tensor of shape [batch, seq, height*sum(fieldsizes), width, depth]
        """
        b, s, w, d, h, c = tensor.shape
        return tensor.permute(0, 1, 4, 5, 2, 3).reshape(b, s, h*c, w, d)
    
    def _to_output_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transforms the output of the model into the desired shape of the output:
        [batch, width, depth, height, sum(fieldsizes)]

        Args:
            tensor (Tensor): Tensor of shape [batch, seq, height*sum(fieldsizes), width, depth]

        Returns:
            Tensor: Transformed tensor of shape [batch, seq, width, depth, height, sum(fieldsizes)]
        """
        if tensor.ndim == 5:
            # tensor is sequence
            b, s = tensor.shape[:2]
            w, d, h = self.out_latent_dims
            
            return tensor.reshape(b, s, h, -1, w, d).permute(0, 1, 4, 5, 2, 3)
        if tensor.ndim == 4:
            # tensor is a single snapshot
            b = tensor.shape[0]
            w, d, h = self.out_latent_dims
            
            return tensor.reshape(b, h, -1, w, d).permute(0, 3, 4, 1, 2)
    
    def summary(self):   
        # LSTM   
        out_shapes = []
        layer_params = []
        if self.use_lstm_encoder:
            for i, cell in enumerate(self.lstm_encoder.cells, 1):
                out_shapes.append((f'EncoderLSTM{i}', [len(cell.hidden_fields), sum(f.size for f in self.field_type), *cell.dims]))
                layer_params.append((f'EncoderLSTM{i}', model_utils.count_trainable_params(cell)))
            
        for i, cell in enumerate(self.lstm_decoder.cells, 1):
            out_shapes.append((f'DecoderLSTM{i}', [len(cell.hidden_fields), sum(f.size for f in self.field_type), *cell.dims]))
            layer_params.append((f'DecoderLSTM{i}', model_utils.count_trainable_params(cell)))
            
        out_shapes.append(('LSTM-Head', [self.latent_channels, sum(f.size for f in self.field_type), *self.latent_dims]))
        layer_params.append((f'LSTM-Head', model_utils.count_trainable_params(self.output_conv)))
        latent_shape = out_shapes[-1][1]
        
        # Autoencoder
        if self.autoencoder is not None:
            ae_out_shapes = list(self.autoencoder.out_shapes.items())
            ae_layer_params = list(self.autoencoder.layer_params.items())
            
            decoder_start = list(self.autoencoder.out_shapes.keys()).index('LatentConv')+1
            
            encoder_out_shapes = ae_out_shapes[:decoder_start]
            decoder_out_shapes = ae_out_shapes[decoder_start:]
            encoder_layer_params = ae_layer_params[:decoder_start]
            decoder_layer_params = ae_layer_params[decoder_start:]
            
            out_shapes = encoder_out_shapes + out_shapes + decoder_out_shapes
            layer_params = encoder_layer_params + layer_params + decoder_layer_params
        
        model_utils.summary(self, OrderedDict(out_shapes), OrderedDict(layer_params), latent_shape)
        
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
        
        batch_size = 2
        warmup_length = 2
        steps = 2

        warmup_input = torch.randn(batch_size, 
                                   warmup_length, 
                                   *self.latent_dims,
                                   sum(field.size for field in self.lstm_decoder.in_fields))
        
        if gpu_device is not None: 
            warmup_input = warmup_input.to(gpu_device)
        
        warmup_input = self._from_input_shape(warmup_input)
        warmup_input = [GeometricTensor(warmup_input[:, t], self.lstm_decoder.in_type) for t in range(warmup_length)]
        
        errors = []
        for el in self.in_type.testing_elements:
            warmup_input_transformed = [x.transform(el) for x in warmup_input]
            
            out1 = self._forward_geometric_latent(warmup_input, steps=steps)
            out2 = self._forward_geometric_latent(warmup_input_transformed, steps=steps)
            
            out1 = torch.stack([y.transform(el).tensor for y in out1], dim=1)
            out2 = torch.stack([y.tensor for y in out2], dim=1)
            
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