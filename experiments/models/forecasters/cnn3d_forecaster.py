import torch
from torch.nn import Module

from layers.conv.conv3d import RB3DConv
from layers.lstm.conv3d_lstm import RB3DConvLSTM

from experiments.models import model_utils
from collections import OrderedDict

#! TODO LayerNorm?
#! parameter initialization
class RB3DForecaster(Module):
    def __init__(
        self,
        autoencoder: torch.nn.Module,
        num_layers: int,
        input_channels: int,
        hidden_channels: list[int],
        latent_dims: tuple[int],
        v_kernel_size: int,
        h_kernel_size: int,
        residual_connection: bool = True,
        nonlinearity = torch.tanh,
        drop_rate: float = 0,
        recurrent_drop_rate: float = 0,
        parallel_ops: bool = True, # applies autoencoder and output layer in parallel (might result in out of memory for large sequences)
        **kwargs
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.latent_dims = latent_dims
        self.residual_connection = residual_connection
        
        self.autoencoder = autoencoder
        self.parallel_ops = parallel_ops
        
        self.lstm = RB3DConvLSTM(num_layers=num_layers, 
                                 input_channels=input_channels, 
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
                                     out_channels=input_channels,
                                     in_dims=latent_dims,
                                     v_kernel_size=v_kernel_size,
                                     h_kernel_size=h_kernel_size,
                                     v_pad_mode='zeros',
                                     h_pad_mode='circular',
                                     bias=True)
        
    def forward(self, warmup_input, steps=1, output_whole_warmup=False):
        # input shape (b,seq,w,d,h,c) or (b,w,d,h,c)
        
        assert warmup_input.ndim==6, "warmup_input must be a sequence"
        
        # encode into latent space
        warmup_latent = self._encode(warmup_input)
            
        output_latent = self.forward_latent(warmup_latent, steps, output_whole_warmup)
        
        # decode into original space
        output = self._decode(output_latent)
        
        return output
    
    def forward_latent(self, warmup_input, steps=1, output_whole_warmup=False):
        # input shape (b,warmup_seq,c,w,d,h) -> (b,forecast_seq,c,w,d,h) or (b,warmup_preds+forecast_seq,c,w,d,h)
        # 2 phases: warmup (lstm gets ground truth as input), autoregression: (lstm gets its own outputs as input, for steps > 1)
        # output_whole_warmup: by default only the states after the warmup are outputted. Set to True to output all 
        # len(warmup_input) predictions wuring warmup rather than just the last one
        warmup_input = self._from_input_shape(warmup_input)
        
        input_channels, *dims = warmup_input.shape[2:]
        assert input_channels == self.input_channels
        assert tuple(dims) == tuple(self.latent_dims)
        
        lstm_autoregressor = self.lstm.autoregress(warmup_input, steps, output_whole_warmup)
        
        lstm_input = None # warmup_input is already provided to LSTM
        outputs = []
        for i in range(steps):
            lstm_out = lstm_autoregressor.send(lstm_input)
            out = self._apply_output_layer(lstm_out)
            
            if self.residual_connection:
                if i == 0:
                    res_input = warmup_input if output_whole_warmup else warmup_input[:, [-1]]
                else:
                    res_input = lstm_input
                out = res_input + out
            
            outputs.append(out)
            lstm_input = out[:, [-1]]
        output = torch.cat(outputs, dim=1)
            
        output = self._to_output_shape(output)
        return output
    
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
    
    def _decode(self, latent):
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
        
        return output
        
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
        return tensor.permute(0, 1, 3, 4, 5, 2)
    
    def summary(self):   
        # LSTM   
        fc_out_shapes = []
        fc_layer_params = []
        for i, cell in enumerate(self.lstm.cells, 1):
            fc_out_shapes.append((f'LSTM{i}', [cell.hidden_channels, *cell.dims]))
            fc_layer_params.append((f'LSTM{i}', model_utils.count_trainable_params(cell)))
            
        fc_out_shapes.append(('LSTM-Head', [self.input_channels, *self.latent_dims]))
        fc_layer_params.append((f'LSTM{i}', model_utils.count_trainable_params(self.output_conv)))
        
        # Autoencoder
        ae_out_shapes = list(self.autoencoder.out_shapes.items())
        ae_layer_params = list(self.autoencoder.layer_params.items())
        
        decoder_start = list(self.autoencoder.out_shapes.keys()).index('LatentConv')+1
        
        encoder_out_shapes = ae_out_shapes[:decoder_start]
        decoder_out_shapes = ae_out_shapes[decoder_start:]
        encoder_layer_params = ae_layer_params[:decoder_start]
        decoder_layer_params = ae_layer_params[decoder_start:]
        
        # Total
        out_shapes = OrderedDict(encoder_out_shapes + fc_out_shapes + decoder_out_shapes)
        layer_params = OrderedDict(encoder_layer_params + fc_layer_params + decoder_layer_params)
        latent_shape = encoder_out_shapes[-1][1]
        
        model_utils.summary(self, out_shapes, layer_params, latent_shape)