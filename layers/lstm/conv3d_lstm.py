import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

from layers.conv.conv3d import RB3DConv

from typing import Literal, Generator


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
        """A convolutional LSTM cell that applies standard 3D convolution (with vertical parameter sharing).

        Args:
            input_channels (int): The number of input channels.
            hidden_channels (int): The number of channels used for the hidden and cell state.
            dims (tuple[int]): The spatial dimensions of the input data.
            v_kernel_size (int): The vertical kernel size.
            h_kernel_size (int): The horizontal kernel size (in both directions).
            v_stride (int, optional): The vertical stride. Defaults to 1.
            h_stride (int, optional): The horizontal stride (in both directions). Defaults to 1.
            v_dilation (int, optional): The vertical dilation. Defaults to 1.
            h_dilation (int, optional): The horizontal dilation. Defaults to 1.
            h_pad_mode (str, optional): The padding applied to the horizontal dimensions. Must be one of the
                following: 'zero', 'circular'. Defaults to 'circular'.
            nonlinearity (optional): The nonlinearity applied to the updates. Defaults to torch.tanh.
            drop_rate (float, optional): The dropout rate applied to the input. Defaults to 0.
            recurrent_drop_rate (float, optional): The dropout rate applied to the hidden state. Defaults to 0.
            bias (bool, optional): Whether to apply a bias after the convolution operations. Defaults to True.
        """
        super().__init__()
        
        assert h_pad_mode in ['zeros', 'circular']
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        self.in_dims = dims
        self.out_dims = dims
        
        self.nonlinearity = nonlinearity
        self.drop_rate = drop_rate
        self.recurrent_drop_rate = recurrent_drop_rate
        self.bias = bias
        
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        
        # computes all three gates in parallel based on the input, hidden state and cell state
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
    
    
    def forward(self, input: Tensor, state: tuple[Tensor]) -> tuple[Tensor]:
        """Computes the next hidden and cell state based on the previous ones and the current input.

        Args:
            input (Tensor): The current input to the cell of shape [batch, channels, width, depth, height].
            state (tuple[Tensor]): The current hidden and cell state.

        Returns:
            tuple[Tensor]: The new hidden and cell state.
        """
        hidden_state, cell_state = state
        
        # apply dropout
        if self.drop_rate > 0 and self.training:
            dropout_mask = self.get_dropout_mask(input)
            input = input * dropout_mask
        if self.recurrent_drop_rate > 0 and self.training:
            recurrent_dropout_mask = self.get_recurrent_dropout_mask(hidden_state)
            hidden_state = hidden_state * recurrent_dropout_mask
        
        # compute gates
        gate_conv_input = torch.cat([input, hidden_state, cell_state], dim=1)
        gate_conv_output = self.gate_conv(gate_conv_input)
        fz, iz, oz, = torch.split(gate_conv_output, self.hidden_channels, dim=1)
        f = torch.sigmoid(fz)
        i = torch.sigmoid(iz)
        o = torch.sigmoid(oz)
        
        # compute cell update
        update_conv_input = torch.cat([input, hidden_state], dim=1)
        cell_update_z = self.cell_update_conv(update_conv_input)
        cell_update = self.nonlinearity(cell_update_z)
        
        # compute new hidden and cell state
        new_cell_state = f * cell_state + i * cell_update
        new_hidden_state = o * self.nonlinearity(new_cell_state)
        
        return new_hidden_state, new_cell_state
    
    
    def init_state(self, batch_size: int, dims: tuple[int]) -> tuple[Tensor]:
        """Initializes the hidden and cell state with zeros.

        Args:
            batch_size (int): The number of samples per batch.
            dims (tuple[int]): The dimensions of the hidden and cell state

        Returns:
            tuple[Tensor]: The initialized hidden and cell state.
        """
        device = next(self.gate_conv.parameters()).device
        
        hidden_state = torch.zeros((batch_size, self.hidden_channels, *dims), device=device)
        cell_state = torch.zeros((batch_size, self.hidden_channels, *dims), device=device)
        
        return hidden_state, cell_state
    
    
    def get_dropout_mask(self, input: Tensor) -> Tensor:
        """Returns the dropout mask which is applied to the input. In case the mask wasn't initialized yet
        or was previously reset, a new one is initialized and stored.

        Args:
            input (Tensor): The current input.

        Returns:
            Tensor: The dropout mask.
        """
        if self._dropout_mask is None:
            self._dropout_mask = F.dropout(torch.ones_like(input), self.drop_rate)
        return self._dropout_mask
    
    
    def get_recurrent_dropout_mask(self, hidden_state: Tensor) -> Tensor:
        """Returns the dropout mask which is applied to the hidden state. In case the mask wasn't initialized yet
        or was previously reset, a new one is initialized and stored.

        Args:
            hidden_state (Tensor): The current hidden state.

        Returns:
            Tensor: The dropout mask.
        """
        if self._recurrent_dropout_mask is None:
            self._recurrent_dropout_mask = F.dropout(torch.ones_like(hidden_state), self.drop_rate)
        return self._recurrent_dropout_mask
    
    
    def reset_dropout_masks(self):
        """Resets the current dropout masks."""
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
        drop_rate: float = 0,
        recurrent_drop_rate: float = 0,
        bias: bool = True
    ):
        """A convolutional LSTM that applies standard 3D convolution (with vertical parameter sharing).

        Args:
            num_layers (int): The number of layers of the LSTM.
            input_channels (int): The number of input channels.
            hidden_channels (list[int] | int): The number of channels used for the hidden and cell state in each layer.
                If a single integer is provided, the same number of hidden channels will be used accross every layer. 
            dims (tuple[int]): The spatial dimensions of the input data.
            v_kernel_size (int): The vertical kernel size.
            h_kernel_size (int): The horizontal kernel size (in both directions).
            v_stride (int, optional): The vertical stride. Defaults to 1.
            h_stride (int, optional): The horizontal stride (in both directions). Defaults to 1.
            v_dilation (int, optional): The vertical dilation. Defaults to 1.
            h_dilation (int, optional): The horizontal dilation. Defaults to 1.
            h_pad_mode (str, optional): The padding applied to the horizontal dimensions. Must be one of the
                following: 'zero', 'circular'. Defaults to 'circular'.
            nonlinearity (optional): The nonlinearity applied to the updates. Defaults to torch.tanh.
            drop_rate (float, optional): The dropout rate applied to the input. Defaults to 0.
            recurrent_drop_rate (float, optional): The dropout rate applied to the hidden state. Defaults to 0.
            bias (bool, optional): Whether to apply a bias after the convolution operations. Defaults to True.
        """
        super().__init__()
        
        if type(hidden_channels) is int:
            hidden_channels = [hidden_channels] * num_layers
        else:
            assert len(hidden_channels) == num_layers
            
        self.num_layers = num_layers
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        self.in_dims = dims
        self.out_dims = dims
        
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
            
            
    def forward(
        self, 
        input: Tensor, 
        state: list[tuple[Tensor]] = None, 
        only_last_output: bool = False
    ) -> tuple[Tensor, list[tuple[Tensor]]]:
        """Forwards an input sequence through the LSTM. Note that this is not autoregressive.

        Args:
            input (Tensor): The input sequence of shape [batch, seq, channels, width, depth, height].
            state (list[tuple[Tensor]], optional): The previous state. If not specified, the state
                is initilized with zeros.
            only_last_output (bool, optional): If set to `True`, only the output w.r.t. the last input will be returned. 
                Defaults to False.

        Returns:
            Tensor, list[tuple[Tensor]]: The output sequence of shape [batch, seq, channels, width, depth, height] 
                (seq will be 1 if only_last_output==True) together with the new state of the LSTM.
        """
        batch_size, series_length, input_channels, *dims = input.shape
        assert input_channels == self.input_channels
        assert tuple(dims) == tuple(self.in_dims)
        
        if state is None:
            state = self.init_state(batch_size, dims)
            
        for cell in self.cells:
            cell.reset_dropout_masks()
            
        # feed the input layer-by-layer through the LSTM
        for i, cell in enumerate(self.cells):
            layer_hidden_states = []
            for t in range(series_length):
                hidden_state, cell_state = cell(input[:, t], state[i])
                state[i] = (hidden_state, cell_state)
                layer_hidden_states.append(hidden_state)
                
            layer_output = torch.stack(layer_hidden_states, dim=1)
            input = layer_output # use the hidden states of the previous layer as input to the next layer
        
        if only_last_output:
            return layer_output[:, [-1]], state
        else:
            return layer_output, state
    
    
    def autoregress(
        self, 
        warmup_input: Tensor, 
        steps: int, 
        state: list[tuple[Tensor]] = None, 
        output_whole_warmup: bool = False
    ) -> Generator[Tensor, Tensor, None]:
        """Generator that autoregressively forwards through the LSTM. First, the whole warmup sequence is 
        forwarded through the network to compute the first output. The next `steps` outputs are then generated
        autoregressively. 
        
        It yields the outputs and expects to be provided with the input for the next prediction via .send(input).
        This allows to apply some prediction head to the LSTM output.

        Args:
            warmup_input (Tensor): The warmup sequence of shape [batch, seq, channels, width, depth, height]
            steps (int): The number of autoregressive steps.
            state (list[tuple[Tensor]], optional): The initial state of the LSTM. Defaults to None.
            output_whole_warmup (bool, optional): If set to `True`, only the output w.r.t. the last element of the
                warmup sequence is yielded. Defaults to False.

        Yields:
            Generator[Tensor, Tensor, None]: The generator yields the outputs and expects to be provided with the input for the 
            next prediction via `.send(input)`.
        """
        input = warmup_input
        for i in range(steps):
            only_last_output = i > 0 or not output_whole_warmup
            output, state = self.forward(input, state=state, only_last_output=only_last_output)
            input = yield output
    
        
    def init_state(self, batch_size: int, dims: tuple[int]) -> list[tuple[Tensor]]:
        """Initializes the hidden and cell states of the layers with zeros.

        Args:
            batch_size (int): The number of samples per batch.
            dims (tuple[int]): The dimensions of the hidden and cell state

        Returns:
            list[tuple[Tensor]]: The initialized hidden and cell states.
        """
        state = []
        for cell in self.cells:
            state.append(cell.init_state(batch_size, dims))
        return state