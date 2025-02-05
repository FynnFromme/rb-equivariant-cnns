import torch
from torch.nn import Module
import math

from layers.conv.conv3d import RB3DConv
from layers.lstm.conv3d_lstm import RB3DConvLSTM

# TODO LayerNorm?
class RB3DConvForecast(Module):
    def __init__(
        self,
        num_layers: int,
        input_channels: int,
        hidden_channels: list[int],
        dims: tuple[int],
        v_kernel_size: int,
        h_kernel_size: int,
        nonlinearity = torch.tanh,
        droprate: float = 0,
        recurrent_droprate: float = 0
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.dims = dims
        
        self.lstm = RB3DConvLSTM(num_layers=num_layers, 
                                 input_channels=input_channels, 
                                 hidden_channels=hidden_channels,
                                 dims=dims, 
                                 v_kernel_size=v_kernel_size, 
                                 h_kernel_size=h_kernel_size, 
                                 nonlinearity=nonlinearity,
                                 droprate=droprate,
                                 recurrent_droprate=recurrent_droprate)
        
        self.dropout = torch.nn.Dropout(droprate)
        
        self.output_conv = RB3DConv(in_channels=hidden_channels[-1],
                                     out_channels=input_channels,
                                     in_dims=dims,
                                     v_kernel_size=v_kernel_size,
                                     h_kernel_size=h_kernel_size,
                                     v_pad_mode='zeros',
                                     h_pad_mode='circular',
                                     bias=True)
        
    def forward(self, input, only_last_output=False):
        # input shape (b,seq,c,w,d,h) or (b,c,w,d,h)
        
        sequence = input.ndim==6
        if not sequence:
            input = input.unsqueeze(1)
            
        batch_size, series_length, input_channels, *dims = input.shape
        assert input_channels == self.input_channels
        assert tuple(dims) == tuple(self.dims)
        
        lstm_out, _ = self.lstm(input, only_last_output=only_last_output)
        
        lstm_out = self.dropout(lstm_out)
        
        output_series_length = 1 if only_last_output else series_length
        output = torch.zeros(batch_size, output_series_length, input_channels, *dims, device=input.device)
        for i in range(output_series_length):
            output[:, i] = self.output_conv(lstm_out[:, i])
            
        return output if sequence else output[:, -1]
    
    def autoregress(self, warmup_input, steps):
        # input shape (b,seq,c,w,d,h)
        batch_size, series_length, input_channels, *dims = warmup_input.shape
        assert input_channels == self.input_channels
        assert tuple(dims) == tuple(self.dims)
        
        lstm_autoregressor = self.lstm.autoregress(warmup_input, steps)
        
        input = None
        output = torch.zeros(batch_size, steps, input_channels, *dims, device=warmup_input.device)
        for i in range(steps):
            lstm_out = lstm_autoregressor.send(input)
            output[:, i] = self.output_conv(lstm_out[:, -1])
            input = output[:, [i]]
            
        return output