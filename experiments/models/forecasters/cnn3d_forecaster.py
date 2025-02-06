import torch
from torch.nn import Module
import math
import numpy as np

from contextlib import nullcontext

from layers.conv.conv3d import RB3DConv
from layers.lstm.conv3d_lstm import RB3DConvLSTM

# TODO LayerNorm?
# TODO Residual Connection?
#! add summary
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
        nonlinearity = torch.tanh,
        drop_rate: float = 0,
        recurrent_drop_rate: float = 0,
        **kwargs
    ):
        super().__init__()
        #! make attribute to switch on/off parallel computation (is more efficient but may result in out of 
        #! memory for large sequences)
        
        self.input_channels = input_channels
        self.latent_dims = latent_dims
        
        self.autoencoder = autoencoder
        
        self.lstm = RB3DConvLSTM(num_layers=num_layers, 
                                 input_channels=input_channels, 
                                 hidden_channels=hidden_channels,
                                 dims=latent_dims, 
                                 v_kernel_size=v_kernel_size, 
                                 h_kernel_size=h_kernel_size, 
                                 nonlinearity=nonlinearity,
                                 drop_rate=drop_rate,
                                 recurrent_drop_rate=recurrent_drop_rate)
        
        self.dropout = torch.nn.Dropout(drop_rate)
        
        self.output_conv = RB3DConv(in_channels=hidden_channels[-1],
                                     out_channels=input_channels,
                                     in_dims=latent_dims,
                                     v_kernel_size=v_kernel_size,
                                     h_kernel_size=h_kernel_size,
                                     v_pad_mode='zeros',
                                     h_pad_mode='circular',
                                     bias=True)
        
    def forward(self, input, only_last_output=True):
        # input shape (b,seq,w,d,h,c) or (b,w,d,h,c)
        
        is_sequence = input.ndim==6
        if not is_sequence:
            input = input.unsqueeze(1)
        
        # encode into latent space
        # Merge batch and sequence dimensions: new shape [B * S, C, W, D, H]
        input_flat = input.reshape(np.prod(input.shape[:2]), *input.shape[2:])
        input_latent_flat = self.autoencoder.encode(input_flat)
        input_latent = input_latent_flat.reshape(*input.shape[:2], *input_latent_flat.shape[1:])
            
        output_latent = self.forward_latent(input_latent, only_last_output)
        
        # decode into original space
        # Merge batch and sequence dimensions: new shape [B * S, C, W, D, H]
        output_latent_flat = output_latent.reshape(np.prod(output_latent.shape[:2]), *output_latent.shape[2:])
        output_flat = self.autoencoder.decode(output_latent_flat)
        output = output_flat.reshape(*output_latent.shape[:2], *output_flat.shape[1:])
        
        return output if is_sequence else output[:, -1]
    
        
    def forward_latent(self, input, only_last_output=False):
        # input shape (b,seq,w,d,h,c) or (b,w,d,h,c)
        
        #! During training only one snapshot is predicted based on input sequence
        #! Would it help during training to integrate all intermediate predictions into the loss function?
        
        is_sequence = input.ndim==6
        if not is_sequence:
            input = input.unsqueeze(1)
        input = self._from_input_shape(input)
            
        batch_size, series_length, input_channels, *dims = input.shape
        assert input_channels == self.input_channels
        assert tuple(dims) == tuple(self.latent_dims)
        
        lstm_out, _ = self.lstm(input, only_last_output=only_last_output)
        lstm_out = self.dropout(lstm_out)
        
        output_series_length, hidden_channels = lstm_out.shape[1:3]
        lstm_out_flat = lstm_out.reshape(batch_size*output_series_length, hidden_channels, *dims)
        
        output_flat = self.output_conv(lstm_out_flat)
        
        output = output_flat.reshape(batch_size, output_series_length, input_channels, *dims)
        
        output = self._to_output_shape(output)
        return output if is_sequence else output[:, -1]
    
    def autoregress(self, warmup_input, steps):
        #! TODO make autoregress and autoregress_latent version
        #! use from_input/to_output 
        # input shape (b,seq,c,w,d,h)
        batch_size, series_length, input_channels, *dims = warmup_input.shape
        assert input_channels == self.input_channels
        assert tuple(dims) == tuple(self.latent_dims)
        
        warmup_input = self._from_input_shape(warmup_input)
        
        lstm_autoregressor = self.lstm.autoregress(warmup_input, steps)
        
        input = None
        output = torch.zeros(batch_size, steps, input_channels, *dims, device=warmup_input.device)
        for i in range(steps):
            lstm_out = lstm_autoregressor.send(input)
            lstm_out = self.dropout(lstm_out)
            output[:, i] = self.output_conv(lstm_out[:, -1])
            input = output[:, [i]]
            
        output = self._to_output_shape(output)
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
        # TODO
        pass