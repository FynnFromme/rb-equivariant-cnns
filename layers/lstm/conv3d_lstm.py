import torch
from torch.nn import Module
from torch.nn import functional as F

from layers.conv.conv3d import RB3DConv

from typing import Literal

class RB3DConvLSTMCell(Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        dims: tuple[int],
        v_kernel_size: int,
        h_kernel_size: int,
        v_stride: int = 1,
        h_stride: int = 1,
        v_dilation: int = 1,
        h_dilation: int = 1,
        h_pad_mode: Literal['zeros', 'circular'] = 'circular',
        nonlinearity = torch.tanh,
        drop_rate: float = 0,
        recurrent_drop_rate: float = 0,
        bias: bool = True
    ):
        super().__init__()
        
        assert h_pad_mode in ['zeros', 'circular']
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.dims = dims
        self.nonlinearity = nonlinearity
        self.drop_rate = drop_rate
        self.recurrent_drop_rate = recurrent_drop_rate
        self.bias = bias
        
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        
        self.gate_conv = RB3DConv(in_channels=input_channels+2*hidden_channels, 
                                  out_channels=3*hidden_channels, 
                                  in_dims=dims,
                                  v_kernel_size=v_kernel_size,
                                  h_kernel_size=h_kernel_size,
                                  v_stride=v_stride,
                                  h_stride=h_stride,
                                  v_dilation=v_dilation,
                                  h_dilation=h_dilation,
                                  v_pad_mode='zeros',
                                  h_pad_mode=h_pad_mode,
                                  bias=bias)
        
        self.cell_update_conv = RB3DConv(in_channels=input_channels+hidden_channels, 
                                         out_channels=hidden_channels, 
                                         in_dims=dims,
                                         v_kernel_size=v_kernel_size,
                                         h_kernel_size=h_kernel_size,
                                         v_stride=v_stride,
                                         h_stride=h_stride,
                                         v_dilation=v_dilation,
                                         h_dilation=h_dilation,
                                         v_pad_mode='zeros',
                                         h_pad_mode=h_pad_mode,
                                         bias=bias)
    
    def forward(self, input: torch.Tensor, state: tuple[torch.Tensor]):
        hidden_state, cell_state = state
        
        if self.drop_rate > 0 and self.training:
            dropout_mask = self.get_dropout_mask(input)
            input = input * dropout_mask
        
        if self.recurrent_drop_rate > 0 and self.training:
            recurrent_dropout_mask = self.get_recurrent_dropout_mask(hidden_state)
            hidden_state = hidden_state * recurrent_dropout_mask
        
        gate_conv_input = torch.cat([input, hidden_state, cell_state], dim=1) # TODO concat at channel dim
        gate_conv_output = self.gate_conv(gate_conv_input)
        fz, iz, oz, = torch.split(gate_conv_output, self.hidden_channels, dim=1) # TODO split at channel dim
        f = torch.sigmoid(fz) #! needs to be equivariant activation for equivariant models
        i = torch.sigmoid(iz) #! needs to be equivariant activation for equivariant models
        o = torch.sigmoid(oz) #! needs to be equivariant activation for equivariant models
        
        update_conv_input = torch.cat([input, hidden_state], dim=1) # TODO concat at channel dim
        cell_update_z = self.cell_update_conv(update_conv_input)
        cell_update = self.nonlinearity(cell_update_z)
        
        new_cell_state = f * cell_state + i * cell_update
        new_hidden_state = o * self.nonlinearity(new_cell_state)
        
        return new_hidden_state, new_cell_state
    
    def init_state(self, batch_size: int, dims: tuple[int]):
        device = next(self.gate_conv.parameters()).device
        hidden_state = torch.zeros((batch_size, self.hidden_channels, *dims), device=device)
        cell_state = torch.zeros((batch_size, self.hidden_channels, *dims), device=device)
        return hidden_state, cell_state
    
    def get_dropout_mask(self, input):
        if self._dropout_mask is None:
            self._dropout_mask = F.dropout(torch.ones_like(input), self.drop_rate) #! needs to be equivariant dropout for equivariant models
        return self._dropout_mask
    
    def get_recurrent_dropout_mask(self, hidden_state):
        if self._recurrent_dropout_mask is None:
            self._recurrent_dropout_mask = F.dropout(torch.ones_like(hidden_state), self.drop_rate) #! needs to be equivariant dropout for equivariant models
        return self._recurrent_dropout_mask
    
    def reset_dropout_masks(self):
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
    
    
class RB3DConvLSTM(Module):
    def __init__(
        self,
        num_layers: int,
        input_channels: int,
        hidden_channels: list[int] | int,
        dims: tuple[int],
        v_kernel_size: int,
        h_kernel_size: int,
        v_stride: int = 1,
        h_stride: int = 1,
        v_dilation: int = 1,
        h_dilation: int = 1,
        h_pad_mode: Literal['zeros', 'circular'] = 'circular',
        nonlinearity = torch.tanh,
        bias: bool = True,
        drop_rate: float = 0,
        recurrent_drop_rate: float = 0
    ):
        super().__init__()
        
        if type(hidden_channels) is int:
            hidden_channels = [hidden_channels] * num_layers
        else:
            assert len(hidden_channels) == num_layers
            
        self.num_layers = num_layers
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.dims = dims
        
        self.cells = torch.nn.ModuleList()
        for i in range(num_layers):
            layer_in_channels = input_channels if i == 0 else hidden_channels[i-1]
            self.cells.append(RB3DConvLSTMCell(input_channels=layer_in_channels, 
                                               hidden_channels=hidden_channels[i],
                                               dims=dims,
                                               v_kernel_size=v_kernel_size,
                                               h_kernel_size=h_kernel_size,
                                               v_stride=v_stride,
                                               h_stride=h_stride,
                                               v_dilation=v_dilation,
                                               h_dilation=h_dilation,
                                               h_pad_mode=h_pad_mode,
                                               nonlinearity=nonlinearity,
                                               bias=bias,
                                               drop_rate=drop_rate,
                                               recurrent_drop_rate=recurrent_drop_rate))
            
            
    def forward(self, input, state=None, only_last_output=False):
        """Note: this is not autoregressive (forced encoding)"""
        # shape (b,seq,c,w,d,h)
        batch_size, series_length, input_channels, *dims = input.shape
        assert input_channels == self.input_channels
        assert tuple(dims) == tuple(self.dims)
        
        if state is None:
            state = self.init_state(batch_size, dims)
            
        for cell in self.cells:
            cell.reset_dropout_masks()
            
        for i, cell in enumerate(self.cells):
            layer_hidden_states = []
            for t in range(series_length):
                hidden_state, cell_state = cell(input[:, t], state[i])
                state[i] = (hidden_state, cell_state)
                layer_hidden_states.append(hidden_state)
            layer_output = torch.stack(layer_hidden_states, dim=1)
            input = layer_output
        
             
        if only_last_output:
            return layer_output[:, [-1]], state
        else:
            return layer_output, state
    
    
    def autoregress(self, warmup_input, steps, output_whole_warmup=False):
        state = None
        input = warmup_input
        for i in range(steps):
            only_last_output = i > 0 or not output_whole_warmup
            output, state = self.forward(input, state=state, only_last_output=only_last_output)
            input = yield output
    
        
    def init_state(self, batch_size: int, dims: tuple[int]):
        state = []
        for cell in self.cells:
            state.append(cell.init_state(batch_size, dims))
        return state