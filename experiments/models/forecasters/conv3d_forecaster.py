import torch
from torch.nn import Module
import numpy as np

from layers.conv.conv3d import RB3DConv
from layers.lstm.conv3d_lstm import RB3DConvLSTM

from experiments.models import model_utils
from collections import OrderedDict

import random

# TODO LayerNorm?
class RB3DLatentForecaster(Module):
    def __init__(
        self,
        num_layers: int,
        latent_channels: int,
        hidden_channels: list[int],
        latent_dims: tuple[int],
        v_kernel_size: int,
        h_kernel_size: int,
        use_lstm_encoder: bool = True,
        residual_connection: bool = True,
        nonlinearity = torch.tanh,
        drop_rate: float = 0,
        recurrent_drop_rate: float = 0,
        min_forced_decoding_prob: float = 0,
        init_forced_decoding_prob: float = 1,
        forced_decoding_epochs: float = 100,
        backprop_through_autoregression: bool = True,
        parallel_ops: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.use_lstm_encoder = use_lstm_encoder
        
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.latent_dims = latent_dims
        self.residual_connection = residual_connection
        
        self.backprop_through_autoregression = backprop_through_autoregression
        self.parallel_ops = parallel_ops
        
        self.min_forced_decoding_prob = min_forced_decoding_prob
        self.init_forced_decoding_prob = init_forced_decoding_prob
        self.forced_decoding_epochs = forced_decoding_epochs
        
        if use_lstm_encoder:
            self.lstm_encoder = RB3DConvLSTM(num_layers=num_layers, 
                                            input_channels=latent_channels, 
                                            hidden_channels=hidden_channels,
                                            dims=latent_dims, 
                                            v_kernel_size=v_kernel_size, 
                                            h_kernel_size=h_kernel_size, 
                                            nonlinearity=nonlinearity,
                                            drop_rate=drop_rate,
                                            recurrent_drop_rate=recurrent_drop_rate,
                                            bias=True)
        else:
            self.lstm_encoder = None
        
        self.lstm_decoder = RB3DConvLSTM(num_layers=num_layers, 
                                         input_channels=latent_channels, 
                                         hidden_channels=hidden_channels,
                                         dims=latent_dims, 
                                         v_kernel_size=v_kernel_size, 
                                         h_kernel_size=h_kernel_size, 
                                         nonlinearity=nonlinearity,
                                         drop_rate=drop_rate,
                                         recurrent_drop_rate=recurrent_drop_rate,
                                         bias=True)
        
        self.dropout = torch.nn.Dropout(drop_rate)
        
        self.output_conv = RB3DConv(in_channels=hidden_channels[-1],
                                    out_channels=latent_channels,
                                    in_dims=latent_dims,
                                    v_kernel_size=v_kernel_size,
                                    h_kernel_size=h_kernel_size,
                                    v_pad_mode='zeros',
                                    h_pad_mode='circular',
                                    bias=True)
        
        self.in_dims = self.lstm_encoder.in_dims if use_lstm_encoder else self.lstm_decoder.in_dims
        self.out_dims = self.output_conv.out_dims
        
    
    def forward(self, warmup_input: torch.Tensor, steps=1, ground_truth=None, epoch=-1):       
        outputs = list(self.forward_gen(warmup_input, steps, ground_truth, epoch))
        return torch.stack(outputs, dim=1)
    
    
    def forward_gen(self, warmup_input, steps=1, ground_truth=None, epoch=-1):
        assert warmup_input.ndim==6, "warmup_input must be a sequence"
                
        # input shape (b,warmup_seq,c,w,d,h) -> (b,forecast_seq,c,w,d,h) or (b,warmup_preds+forecast_seq,c,w,d,h)
        # 2 phases: warmup (lstm gets ground truth as input), autoregression: (lstm gets its own outputs as input, for steps > 1)
        warmup_input = self._from_input_shape(warmup_input)
        
        if ground_truth is not None:
            ground_truth = self._from_input_shape(ground_truth)
            assert ground_truth.size(1) == steps
        
        input_channels, *dims = warmup_input.shape[2:]
        assert input_channels == self.latent_channels
        assert tuple(dims) == tuple(self.latent_dims)
        
        if self.use_lstm_encoder:
            _, encoded_state = self.lstm_encoder.forward(warmup_input)
            decoder_input = warmup_input[:, [-1]]
        else:
            encoded_state = None
            decoder_input = warmup_input
        
        lstm_autoregressor = self.lstm_decoder.autoregress(decoder_input, steps, encoded_state)
        
        lstm_input = None # decoder_input is already provided to LSTM
        for i in range(steps):
            lstm_out = lstm_autoregressor.send(lstm_input)
            out = self._apply_output_layer(lstm_out)
            
            if self.residual_connection:
                res_input = decoder_input[:, [-1]] if i == 0 else lstm_input
                out = res_input + out
            
            yield self._to_output_shape(out[:, -1])
            
            forced_decoding_prob = self._forced_decoding_prob(ground_truth, epoch)
            forced_decoding = random.random() < forced_decoding_prob

            if forced_decoding:
                # use ground truth as input
                lstm_input = ground_truth[:, [i]]
            else:
                # autoregressive prediction
                lstm_input = out[:, [-1]]
                if not self.backprop_through_autoregression:
                    lstm_input = lstm_input.detach()
                    
    
    def _apply_output_layer(self, lstm_out):  
        # shape: (b,seq,c,w,d,h)      
        batch_size, seq_length = lstm_out.shape[:2]
        
        if self.parallel_ops:
            # apply output layer in parallel to whole sequence
            lstm_out_flat = lstm_out.reshape(batch_size*seq_length, *lstm_out.shape[2:])
            
            lstm_out_flat = self.dropout(lstm_out_flat)
            output_flat = self.output_conv(lstm_out_flat)
            
            output = output_flat.reshape(batch_size, seq_length, *output_flat.shape[1:])
        else:
            outputs = []
            for i in range(seq_length):
                hidden_state = lstm_out[:, i]
                hidden_state = self.dropout(hidden_state)
                outputs.append(self.output_conv(hidden_state))
            output = torch.stack(outputs, dim=1)
                   
        return output
    
    
    def _forced_decoding_prob(self, ground_truth, epoch):
        if not self.training:
            return 0
        if epoch < 0 or ground_truth is None:
            return 0
        
        return max(self.min_forced_decoding_prob, self.init_forced_decoding_prob * (1 - (epoch-1)/self.forced_decoding_epochs))     

        
    def _from_input_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transforms an input tensor of shape [batch, sequence, width, depth, height, channels] into the
        shape required for this model.

        Args:
            tensor (Tensor): Tensor of shape [batch, sequence, width, depth, height, channels].

        Returns:
            Tensor: Transformed tensor of shape [batch, sequence, channels, width, depth, height]
        """
        return tensor.permute(0, 1, 5, 2, 3, 4)
    
    
    def _to_output_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transforms the output of the model into the desired shape of the output:
        [batch, sequence, width, depth, height, channels]

        Args:
            tensor (Tensor): Tensor of shape [batch, sequence, channels, width, depth, height].

        Returns:
            Tensor: Transformed tensor of shape [batch, sequence, width, depth, height, channels]
        """
        if tensor.ndim == 6:
            # is sequence
            return tensor.permute(0, 1, 3, 4, 5, 2)
        else:
            # is a single snapshot
            return tensor.permute(0, 2, 3, 4, 1)
        
        
    def layer_out_shapes(self) -> OrderedDict:
        out_shapes = OrderedDict()
        
        if self.use_lstm_encoder:
            for i, cell in enumerate(self.lstm_encoder.cells, 1):
                out_shapes[f'EncoderLSTM{i}'] = [cell.hidden_channels, *cell.out_dims]
            
        for i, cell in enumerate(self.lstm_decoder.cells, 1):
            out_shapes[f'DecoderLSTM{i}'] = [cell.hidden_channels, *cell.out_dims]
            
        out_shapes['LSTM-Head'] = [self.latent_channels, *self.latent_dims]
        
        return out_shapes
        
        
    def layer_params(self) -> OrderedDict:
        layer_params = OrderedDict()
        
        if self.use_lstm_encoder:
            for i, cell in enumerate(self.lstm_encoder.cells, 1):
                layer_params[f'EncoderLSTM{i}'] = model_utils.count_trainable_params(cell)
            
        for i, cell in enumerate(self.lstm_decoder.cells, 1):
            layer_params[f'DecoderLSTM{i}'] = model_utils.count_trainable_params(cell)
            
        layer_params[f'LSTM-Head'] = model_utils.count_trainable_params(self.output_conv)
        
        return layer_params
    
    
    def summary(self):   
        out_shapes = self.layer_out_shapes()
        params = self.layer_params()
        
        model_utils.summary(self, out_shapes, params, steerable=False)
        
        
class RB3DForecaster(Module):
    def __init__(self,
                 latent_forecaster: RB3DLatentForecaster,
                 autoencoder: torch.nn.Module,
                 train_autoencoder: bool = False,
                 parallel_ops: bool = True, # applies autoencoder in parallel (might result in out of memory for large sequences)
                 **kwargs):
        super().__init__()
        
        self.latent_forecaster = latent_forecaster
        self.autoencoder = autoencoder
        
        self.train_autoencoder = train_autoencoder
        self.parallel_ops = parallel_ops
        
        
        self.in_dims, self.out_dims = self.autoencoder.in_dims, self.autoencoder.out_dims
        
    
    def forward(self, warmup_input, steps=1, ground_truth=None, epoch=-1):
        # input shape (b,warmup_seq,c,w,d,h) -> (b,forecast_seq,c,w,d,h) or (b,warmup_preds+forecast_seq,c,w,d,h)
        # 2 phases: warmup (lstm gets ground truth as input), autoregression: (lstm gets its own outputs as input, for steps > 1)
        # output_whole_warmup: by default only the states after the warmup are outputted. Set to True to output all 
        # len(warmup_input) predictions wuring warmup rather than just the last one
        
        # does not use the forward_gen generator in order to be able to apply the decoder in parallel to
        # the whole sequence
        
        assert warmup_input.ndim==6, "warmup_input must be a sequence"
        
        # encode into latent space
        warmup_latent = self._encode_to_latent(warmup_input)
    
        if not self.train_autoencoder:
            warmup_latent = warmup_latent.detach()
            
        output_latent = self.latent_forecaster.forward(warmup_latent, steps, ground_truth, epoch)
        
        # decode into original space
        return self._decode_from_latent(output_latent)
    
    
    def forward_gen(self, warmup_input, steps=1, ground_truth=None, epoch=-1):
        # input shape (b,warmup_seq,c,w,d,h) -> (b,forecast_seq,c,w,d,h) or (b,warmup_preds+forecast_seq,c,w,d,h)
        # 2 phases: warmup (lstm gets ground truth as input), autoregression: (lstm gets its own outputs as input, for steps > 1)
        # output_whole_warmup: by default only the states after the warmup are outputted. Set to True to output all 
        # len(warmup_input) predictions wuring warmup rather than just the last one
        
        assert warmup_input.ndim==6, "warmup_input must be a sequence"
        
        # encode into latent space
        warmup_latent = self._encode_to_latent(warmup_input)
    
        if not self.train_autoencoder:
            warmup_latent = warmup_latent.detach()
            
        for output_latent in self.latent_forecaster.forward_gen(warmup_latent, steps, ground_truth, epoch):
            # decode into original space
            output = self._decode_from_latent(output_latent)
            yield output
            
    
    def _encode_to_latent(self, input):
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
    
    
    def _decode_from_latent(self, latent):
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
    
    
    def summary(self):   
        # Forecaster
        latent_out_shapes = self.latent_forecaster.layer_out_shapes()
        latent_params = self.latent_forecaster.layer_params()
        
        # Autoencoder
        encoder_out_shapes = self.autoencoder.layer_out_shapes('encoder')
        decoder_out_shapes = self.autoencoder.layer_out_shapes('decoder')
        encoder_layer_params = self.autoencoder.layer_params('encoder')
        decoder_layer_params = self.autoencoder.layer_params('decoder')
        
        out_shapes = encoder_out_shapes | latent_out_shapes | decoder_out_shapes
        layer_params = encoder_layer_params | latent_params | decoder_layer_params
        
        model_utils.summary(self, out_shapes, layer_params, steerable=False)
        
        print(f'\nShape of latent space: {encoder_out_shapes["LatentConv"]}')
    
        print(f'\nLatent-Input-Ratio: {np.prod(self.autoencoder.latent_shape)/np.prod(encoder_out_shapes["Input"])*100:.2f}%')