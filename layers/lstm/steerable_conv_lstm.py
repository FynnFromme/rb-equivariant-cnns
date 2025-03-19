import torch
from torch import Tensor
from torch.nn import functional as F

from escnn import nn as enn
from escnn.nn import FieldType, GeometricTensor
from escnn.gspaces import GSpace
from escnn.group import Representation

from layers.conv.steerable_conv import RBSteerableConv

from typing import Literal, Generator


class RBSteerableConvLSTMCell(enn.EquivariantModule):
    def __init__(
        self,
        gspace: GSpace, 
        in_fields: list[Representation], 
        hidden_fields: list[Representation], 
        dims: tuple[int],
        v_kernel_size: int,
        h_kernel_size: int,
        v_stride: int = 1,
        h_stride: int = 1,
        h_dilation: int = 1,
        h_pad_mode: Literal['zero', 'circular'] = 'circular',
        nonlinearity: Literal['relu', 'elu', 'tanh'] = 'tanh',
        drop_rate: float = 0,
        recurrent_drop_rate: float = 0,
        bias: bool = True
    ):
        """A convolutional LSTM cell that applies steerable convolutions with 3D kernels that are not
        shared vertically due to RB not being vertically translation equivariant.

        Args:
            gspace (GSpace): The group of transformations to be equivariant to. For `gspaces.flipRot2dOnR2(N)`
                the layer is equivariant to horizontal flips and rotations. Use `gspaces.rot2dOnR2(N)` for only
                rotational equivariance.
            in_fields (list[Representation]): The fields of the layer's input. This corresponds to input channels
                in standard convolutions.
            hidden_fields (list[Representation]): The fields of the hidden and cell state.
            dims (tuple[int]): The spatial dimensions of the input data.
            v_kernel_size (int): The vertical kernel size.
            h_kernel_size (int): The horizontal kernel size (in both directions).
            v_stride (int, optional): The vertical stride. Defaults to 1.
            h_stride (int, optional): The horizontal stride (in both directions). Defaults to 1.
            v_dilation (int, optional): The vertical dilation. Defaults to 1.
            h_dilation (int, optional): The horizontal dilation. Defaults to 1.
            h_pad_mode (str, optional): The padding applied to the horizontal dimensions. Must be one of the
                following: 'zero', 'circular'. Defaults to 'circular'.
            nonlinearity (str, optional): The nonlinearity applied to the updates. Must be one of the following:
                'relu', 'elu', 'tanh'. Defaults to 'tanh'.
            drop_rate (float, optional): The dropout rate applied to the input. Defaults to 0.
            recurrent_drop_rate (float, optional): The dropout rate applied to the hidden state. Defaults to 0.
            bias (bool, optional): Whether to apply a bias after the convolution operations. Defaults to True.
        """
        super().__init__()
        
        assert h_pad_mode in ['zero', 'circular']
        
        self.gspace = gspace
        self.in_fields = in_fields
        self.out_fields = in_fields
        self.hidden_fields = hidden_fields
        
        self.in_dims = dims
        self.out_dims = dims
        
        self.in_type = FieldType(gspace, self.in_dims[-1]*self.in_fields)
        self.hidden_type = FieldType(gspace, self.in_dims[-1]*self.hidden_fields)
        self.out_type = self.hidden_type
        
        self.nonlinearity = enn.PointwiseNonLinearity(self.hidden_type, f'p_{nonlinearity}')
        self.sigmoid = enn.PointwiseNonLinearity(self.hidden_type, f'p_sigmoid')
        self.drop_rate = drop_rate
        self.recurrent_drop_rate = recurrent_drop_rate
        self.bias = bias
        
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        
        # computes all three gates in parallel based on the input, hidden state and cell state
        self.gate_conv = RBSteerableConv(gspace=gspace,
                                         in_fields=in_fields+2*hidden_fields, 
                                         out_fields=3*hidden_fields, 
                                         in_dims=dims,
                                         v_kernel_size=v_kernel_size,
                                         h_kernel_size=h_kernel_size,
                                         v_stride=v_stride,
                                         h_stride=h_stride,
                                         h_dilation=h_dilation,
                                         v_pad_mode='zero',
                                         h_pad_mode=h_pad_mode,
                                         bias=bias)
        
        self.cell_update_conv = RBSteerableConv(gspace=gspace,
                                                in_fields=in_fields+hidden_fields, 
                                                out_fields=hidden_fields, 
                                                in_dims=dims,
                                                v_kernel_size=v_kernel_size,
                                                h_kernel_size=h_kernel_size,
                                                v_stride=v_stride,
                                                h_stride=h_stride,
                                                h_dilation=h_dilation,
                                                v_pad_mode='zero',
                                                h_pad_mode=h_pad_mode,
                                                bias=bias)


    def forward(self, input: GeometricTensor, state: tuple[GeometricTensor]) -> tuple[GeometricTensor]:
        """Computes the next hidden and cell state based on the previous ones and the current input.

        Args:
            input (GeometricTensor): The current input to the cell of shape [batch, height*sum(fieldsizes), width, depth].
            state (tuple[GeometricTensor]): The current hidden and cell state.

        Returns:
            tuple[GeometricTensor]: The new hidden and cell state.
        """
        hidden_state, cell_state = state
        
        # apply dropout - note: this is pointwise dropout and is only suited for fields that support pointwise operations
        if self.drop_rate > 0 and self.training:
            dropout_mask = self.get_dropout_mask(input)
            input = GeometricTensor(input.tensor * dropout_mask, input.type)
        if self.recurrent_drop_rate > 0 and self.training:
            recurrent_dropout_mask = self.get_recurrent_dropout_mask(hidden_state)
            hidden_state = GeometricTensor(hidden_state.tensor * recurrent_dropout_mask, hidden_state.type)
        
        # compute gates
        gate_conv_input = self._concat_fields([(input, self.in_fields), (hidden_state, self.hidden_fields), (cell_state, self.hidden_fields)])
        gate_conv_output = self.gate_conv(gate_conv_input)
        fz, iz, oz, = self._split_fields(gate_conv_output, self.hidden_fields)
        f = self.sigmoid(fz)
        i = self.sigmoid(iz)
        o = self.sigmoid(oz)
        
        # compute cell update
        update_conv_input = self._concat_fields([(input, self.in_fields), (hidden_state, self.hidden_fields)])
        cell_update_z = self.cell_update_conv(update_conv_input)
        cell_update = self.nonlinearity(cell_update_z)
        
        # compute new hidden and cell state
        new_cell_state = GeometricTensor(f.tensor * cell_state.tensor + i.tensor * cell_update.tensor, cell_state.type)
        new_hidden_state = GeometricTensor(o.tensor * self.nonlinearity(new_cell_state).tensor, hidden_state.type)
        
        return new_hidden_state, new_cell_state
    
    
    def _concat_fields(self, geometric_tensors: list[tuple[GeometricTensor, list[Representation]]]) -> GeometricTensor:
        """Concatenates the fields of multiple geometric tensors of shape [batch, height*sum(fieldsizes), width, depth].
        This is equivalent to concatenating channels of conventional convolution inputs.

        Args:
            geometric_tensors (list[tuple[GeometricTensor, list[Representation]]]): A list of pairs, each consisting of
                a geometric tensor of shape [batch, height*sum(fieldsizes), width, depth] and a list of fields of that tensor.

        Returns:
            GeometricTensor: A geometric tensor with concatenated fields of shape [batch, height*sum(fieldsizes'), width, depth].
        """
        tensors = []
        output_fields = []
        for geom_tensor, fields in geometric_tensors:
            # split height and field dimension
            tensor = geom_tensor.tensor.reshape(-1, self.in_dims[-1], sum(field.size for field in fields), *self.in_dims[:2])
            
            tensors.append(tensor)
            output_fields.extend(fields)
            
        output = torch.cat(tensors, dim=2) # concat fields
        
         # combine height and field dimension
        output = output.reshape(-1, self.in_dims[-1]*sum(field.size for field in output_fields), *self.in_dims[:2])
        
        return GeometricTensor(output, FieldType(self.gspace, self.in_dims[-1]*output_fields))
        
        
    def _split_fields(self, geometric_tensor: GeometricTensor, fields: list[Representation]) -> tuple[GeometricTensor]:
        """Splits a geometric tensor of shape [batch, height*sum(fieldsizes), width, depth] along the field dimension into
        multiple geometric tensors.

        Args:
            geometric_tensor (GeometricTensor): The geometric tensor of shape [batch, height*sum(fieldsizes), width, depth] to split.
            fields (list[Representation]): The list of fields each of the resulting geometric tensors should have.
                This assumes that the input tensor can be split into n tensors with these fields.

        Returns:
            tuple[GeometricTensor]: The resulting geometric tensors of shape [batch, height*sum(fieldsizes'), width, depth]
        """
        # split height and field dimension
        batch_size = geometric_tensor.shape[0]
        tensor = geometric_tensor.tensor.reshape(batch_size, self.in_dims[-1], -1, *self.in_dims[:2])
            
        channels = sum(field.size for field in fields)
        output_tensors = torch.split(tensor, channels, dim=2)
        
        output_geom_tensors = []
        for output_tensor in output_tensors:
            output_tensor = output_tensor.reshape(batch_size, self.in_dims[-1]*channels, *self.in_dims[:2])
            output_geom_tensors.append(GeometricTensor(output_tensor, FieldType(self.gspace, self.in_dims[-1]*fields)))
        
        return output_geom_tensors
    
    
    def init_state(self, batch_size: int, dims: tuple[int]) -> tuple[GeometricTensor]:
        """Initializes the hidden and cell state with zeros.

        Args:
            batch_size (int): The number of samples per batch.
            dims (tuple[int]): The dimensions of the hidden and cell state

        Returns:
            tuple[GeometricTensor]: The initialized hidden and cell state.
        """
        device = next(self.gate_conv.parameters()).device
        hidden_state = torch.zeros((batch_size, self.hidden_type.size, *dims[:2]), device=device)
        cell_state = torch.zeros((batch_size, self.hidden_type.size, *dims[:2]), device=device)
        
        hidden_state = GeometricTensor(hidden_state, self.hidden_type)
        cell_state = GeometricTensor(cell_state, self.hidden_type)
        
        return hidden_state, cell_state
    
    
    def get_dropout_mask(self, input: GeometricTensor) -> Tensor:
        """Returns the dropout mask which is applied to the input. In case the mask wasn't initialized yet
        or was previously reset, a new one is initialized and stored.

        Args:
            input (GeometricTensor): The current input.

        Returns:
            Tensor: The dropout mask.
        """
        if self._dropout_mask is None:
            self._dropout_mask = F.dropout(torch.ones_like(input.tensor), self.drop_rate)
        return self._dropout_mask
    
    
    def get_recurrent_dropout_mask(self, hidden_state: GeometricTensor) -> Tensor:
        """Returns the dropout mask which is applied to the hidden state. In case the mask wasn't initialized yet
        or was previously reset, a new one is initialized and stored.

        Args:
            hidden_state (GeometricTensor): The current hidden state.

        Returns:
            Tensor: The dropout mask.
        """
        if self._recurrent_dropout_mask is None:
            self._recurrent_dropout_mask = F.dropout(torch.ones_like(hidden_state.tensor), self.drop_rate)
        return self._recurrent_dropout_mask
    
    
    def reset_dropout_masks(self):
        """Resets the current dropout masks."""
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        
        
    def evaluate_output_shape(self, input_shape: tuple) -> tuple:
        """Compute the shape the output tensor which would be generated by this module when a tensor with shape
        ``input_shape`` is provided as input.
        
        Args:
            input_shape (tuple): shape of the input tensor

        Returns:
            shape of the output tensor
            
        """
        b, _, w, d = input_shape
        
        return (b, self.hidden_type.size, w, d)
    
    
class RBSteerableConvLSTM(enn.EquivariantModule):
    def __init__(
        self,
        gspace: GSpace,
        num_layers: int,
        in_fields: list[Representation], 
        hidden_fields: list[list[Representation]] | list[Representation], 
        dims: tuple[int],
        v_kernel_size: int,
        h_kernel_size: int,
        v_stride: int = 1,
        h_stride: int = 1,
        h_dilation: int = 1,
        h_pad_mode: Literal['zero', 'circular'] = 'circular',
        nonlinearity: Literal['relu', 'elu', 'tanh'] = 'tanh',
        bias: bool = True,
        drop_rate: float = 0,
        recurrent_drop_rate: float = 0
    ):
        """A convolutional LSTM that applies steerable convolutions with 3D kernels that are not
        shared vertically due to RB not being vertically translation equivariant.

        Args:
            gspace (GSpace): The group of transformations to be equivariant to. For `gspaces.flipRot2dOnR2(N)`
                the layer is equivariant to horizontal flips and rotations. Use `gspaces.rot2dOnR2(N)` for only
                rotational equivariance.
            num_layers (int): The number of layers of the LSTM.
            in_fields (list[Representation]): The fields of the layer's input. This corresponds to input channels
                in standard convolutions.
            hidden_fields (list[list[Representation]] | list[Representation]): The fields used for the hidden and cell state in 
                each layer. If a single list of fields is provided, the same fields will be used accross every layer. 
            dims (tuple[int]): The spatial dimensions of the input data.
            v_kernel_size (int): The vertical kernel size.
            h_kernel_size (int): The horizontal kernel size (in both directions).
            v_stride (int, optional): The vertical stride. Defaults to 1.
            h_stride (int, optional): The horizontal stride (in both directions). Defaults to 1.
            v_dilation (int, optional): The vertical dilation. Defaults to 1.
            h_dilation (int, optional): The horizontal dilation. Defaults to 1.
            h_pad_mode (str, optional): The padding applied to the horizontal dimensions. Must be one of the
                following: 'zero', 'circular'. Defaults to 'circular'.
            nonlinearity (str, optional): The nonlinearity applied to the updates. Must be one of the following:
                'relu', 'elu', 'tanh'. Defaults to 'tanh'.
            drop_rate (float, optional): The dropout rate applied to the input. Defaults to 0.
            recurrent_drop_rate (float, optional): The dropout rate applied to the hidden state. Defaults to 0.
            bias (bool, optional): Whether to apply a bias after the convolution operations. Defaults to True.
        """
        super().__init__()
        
        if type(hidden_fields[0]) is list:
            assert len(hidden_fields) == num_layers
        else:
            hidden_fields = [hidden_fields] * num_layers
            
        self.gspace = gspace
        self.in_fields = in_fields
        self.hidden_fields = hidden_fields
        
        self.in_dims = dims
        self.out_dims = dims
        
        self.in_type = FieldType(gspace, self.in_dims[-1]*self.in_fields)
        self.hidden_types = [FieldType(gspace, self.in_dims[-1]*hf) for hf in hidden_fields]
        self.out_type = self.hidden_types[-1]
            
        self.num_layers = num_layers
        self.dims = dims
        
        self.cells = torch.nn.ModuleList()
        for i in range(num_layers):
            layer_in_fields = in_fields if i == 0 else hidden_fields[i-1]
            self.cells.append(RBSteerableConvLSTMCell(gspace=gspace,
                                                      in_fields=layer_in_fields, 
                                                      hidden_fields=hidden_fields[i],
                                                      dims=dims,
                                                      v_kernel_size=v_kernel_size,
                                                      h_kernel_size=h_kernel_size,
                                                      v_stride=v_stride,
                                                      h_stride=h_stride,
                                                      h_dilation=h_dilation,
                                                      h_pad_mode=h_pad_mode,
                                                      nonlinearity=nonlinearity,
                                                      bias=bias,
                                                      drop_rate=drop_rate,
                                                      recurrent_drop_rate=recurrent_drop_rate))
            
            
    def forward(
        self, 
        input: list[GeometricTensor], 
        state: list[tuple[GeometricTensor]] = None, 
        only_last_output: bool = False
    ) -> tuple[list[GeometricTensor], list[tuple[GeometricTensor]]]:
        """Forwards an input sequence through the LSTM. Note that this is not autoregressive.

        Args:
            input (list[GeometricTensor]): The input sequence of geometric tensors, each of shape 
                [batch, inHeight*sum(inFieldsizes), inWidth, inDepth].
            state (list[tuple[GeometricTensor]], optional): The previous state. If not specified, the state
                is initilized with zeros.
            only_last_output (bool, optional): If set to `True`, only the output w.r.t. the last input will be returned. 
                Defaults to False.

        Returns:
            list[GeometricTensor], list[tuple[GeometricTensor]]: The output sequence of geometric tensors, each of shape 
                [batch, inHeight*sum(inFieldsizes), inWidth, inDepth] together with the new state of the LSTM.
        """        
        seq_length = len(input)
        batch_size, _, width, depth = input[0].shape
        assert input[0].type == self.in_type
        assert (width, depth) == tuple(self.dims[:2])
        
        if state is None:
            state = self.init_state(batch_size, (self.dims))
            
        for cell in self.cells:
            cell.reset_dropout_masks()
            
        # feed the input layer-by-layer through the LSTM
        for i, cell in enumerate(self.cells):
            layer_hidden_states = []
            for t in range(seq_length):
                hidden_state, cell_state = cell(input[t], state[i])
                state[i] = (hidden_state, cell_state)
                layer_hidden_states.append(hidden_state)
            
            input = layer_hidden_states # use the hidden states of the previous layer as input to the next layer
             
        if only_last_output:
            return [layer_hidden_states[-1]], state
        else:
            return layer_hidden_states, state
        
    
    def autoregress(
        self, 
        warmup_input: list[GeometricTensor], 
        steps: int, 
        state: list[tuple[GeometricTensor]] = None, 
        output_whole_warmup: bool = False
    ) -> Generator[list[GeometricTensor], list[GeometricTensor], None]:
        """Generator that autoregressively forwards through the LSTM. First, the whole warmup sequence is 
        forwarded through the network to compute the first output. The next `steps` outputs are then generated
        autoregressively. 
        
        It yields the outputs and expects to be provided with the input for the next prediction via .send(input).
        This allows to apply some prediction head to the LSTM output.

        Args:
            warmup_input (list[GeometricTensor]): The warmup sequence of geometric tensors, each of shape
                [batch, inHeight*sum(inFieldsizes), inWidth, inDepth].
            steps (int): The number of autoregressive steps.
            state (list[tuple[GeometricTensor]], optional): The initial state of the LSTM. Defaults to None.
            output_whole_warmup (bool, optional): If set to `True`, only the output w.r.t. the last element of the
                warmup sequence is yielded. Defaults to False.

        Yields:
            Generator[list[GeometricTensor], list[GeometricTensor], None]: The generator yields the outputs and expects 
            to be provided with the input for the next prediction via `.send(input)`.
        """
        input = warmup_input
        for i in range(steps):
            only_last_output = i > 0 or not output_whole_warmup
            output, state = self.forward(input, state=state, only_last_output=only_last_output)
            input = yield output
    
        
    def init_state(self, batch_size: int, dims: tuple[int]) -> list[tuple[GeometricTensor]]:
        """Initializes the hidden and cell states of the layers with zeros.

        Args:
            batch_size (int): The number of samples per batch.
            dims (tuple[int]): The dimensions of the hidden and cell state

        Returns:
            list[tuple[GeometricTensor]]: The initialized hidden and cell states.
        """
        state = []
        for cell in self.cells:
            state.append(cell.init_state(batch_size, dims))
        return state
    
    
    def evaluate_output_shape(self, input_shape: tuple) -> tuple:
        """Compute the shape the output tensor which would be generated by this module when a tensor with shape
        ``input_shape`` is provided as input.
        
        Args:
            input_shape (tuple): shape of the input tensor

        Returns:
            shape of the output tensor
            
        """
        b, _, w, d = input_shape
        
        return (b, self.cells[-1].hidden_type.size, w, d)