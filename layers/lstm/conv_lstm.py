import torch
from torch.nn import Module

class ConvLSTMCell(Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size,
        nonlinearity,
        bias
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.nonlinearity = nonlinearity
        self.bias = bias
        
        # TODO use our specialized convs with padding
        self.gate_conv = torch.nn.Conv3d(in_channels=input_channels+2*hidden_channels, 
                                    out_channels=3*hidden_channels, 
                                    kernel_size=kernel_size,
                                    bias=bias,
                                    padding='same')
        self.cell_update_conv = torch.nn.Conv3d(in_channels=input_channels+hidden_channels, 
                                    out_channels=hidden_channels, 
                                    kernel_size=kernel_size,
                                    bias=bias,
                                    padding='same')
    
    def forward(self, input: torch.Tensor, state: tuple[torch.Tensor]):
        hidden_state, cell_state = state
        gate_conv_input = torch.cat([input, hidden_state, cell_state], dim=1) # TODO concat at channel dim
        gate_conv_output = self.gate_conv(gate_conv_input)
        fz, iz, oz, = torch.split(gate_conv_output, self.hidden_channels, dim=1) # TODO split at channel dim
        f = torch.sigmoid(fz)
        i = torch.sigmoid(iz)
        o = torch.sigmoid(oz)
        
        update_conv_input = torch.cat([input, hidden_state], dim=1) # TODO concat at channel dim
        cell_update_z = self.cell_update_conv(update_conv_input)
        cell_update = self.nonlinearity(cell_update_z)
        
        new_cell_state = f * cell_state + i * cell_update
        new_hidden_state = o * self.nonlinearity(new_cell_state)
        
        return new_hidden_state, new_cell_state
    
    def init_state(self, batch_size: int, dims: tuple[int]):
        device = self.gate_conv.weight.device
        hidden_state = torch.zeros((batch_size, self.hidden_channels, *dims), device=device)
        cell_state = torch.zeros((batch_size, self.hidden_channels, *dims), device=device)
        return hidden_state, cell_state
    
    
class ConvLSTM(Module):
    def __init__(
        self,
        num_layers: int,
        input_channels: int,
        hidden_channels: list[int] | int,
        kernel_size: int = 3,
        nonlinearity = torch.tanh,
        bias: bool = True
    ):
        super().__init__()
        
        if type(hidden_channels) is int:
            hidden_channels = [hidden_channels] * num_layers
            
        self.num_layers = num_layers
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        self.cells = []
        for i in range(num_layers):
            layer_in_channels = input_channels if i == 0 else hidden_channels[i-1]
            self.cells.append(ConvLSTMCell(layer_in_channels, hidden_channels[i], kernel_size, nonlinearity, bias))
            
            
    def forward(self, input, state=None):
        batch_size, series_length, input_channels, *dims = input.shape
        assert input_channels == self.input_channels
        
        if state is None:
            state = self.init_state(batch_size, dims)
            
        for i, cell in enumerate(self.cells):
            layer_hidden_states = []
            for t in range(series_length):
                hidden_state, cell_state = cell(input[:, t], state[i])
                state[i] = (hidden_state, cell_state)
                layer_hidden_states.append(hidden_state)
            layer_output = torch.stack(layer_hidden_states, dim=1)
            input = layer_output
             
        return layer_output, state
    
        
    def init_state(self, batch_size: int, dims: tuple[int]):
        state = []
        for cell in self.cells:
            state.append(cell.init_state(batch_size, dims))
        return state