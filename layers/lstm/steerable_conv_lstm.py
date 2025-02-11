import torch
from torch.nn import functional as F

from layers.conv.steerable_conv import RBSteerableConv

from escnn import nn as enn
from escnn.nn import FieldType, GeometricTensor
from escnn.gspaces import GSpace
from escnn.group import Representation

from typing import Literal

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
        super().__init__()
        
        assert h_pad_mode in ['zero', 'circular']
        
        self.gspace = gspace
        self.in_fields = in_fields
        self.out_fields = in_fields
        self.hidden_fields = hidden_fields
        
        self.in_height = dims[-1]
        self.out_height = self.in_height
        
        self.in_dims = dims
        self.out_dims = dims
        
        self.in_type = FieldType(gspace, self.in_height*self.in_fields)
        self.out_type = self.in_type
        self.hidden_type = FieldType(gspace, self.in_height*self.hidden_fields)
        
        self.dims = dims
        self.nonlinearity = enn.PointwiseNonLinearity(self.hidden_type, f'p_{nonlinearity}')
        self.sigmoid = enn.PointwiseNonLinearity(self.hidden_type, f'p_sigmoid')
        self.drop_rate = drop_rate
        self.recurrent_drop_rate = recurrent_drop_rate
        self.bias = bias
        
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        
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

    def forward(self, input: enn.GeometricTensor, state: tuple[enn.GeometricTensor]):
        # shape [batch, inHeight*sum(inFieldsizes), inWidth, inDepth]
        hidden_state, cell_state = state
        
        # apply dropout - note: this is pointwise dropout and is only suited for fields that support pointwise operations
        if self.drop_rate > 0 and self.training:
            dropout_mask = self.get_dropout_mask(input)
            input = GeometricTensor(input.tensor * dropout_mask, input.type)
        if self.recurrent_drop_rate > 0 and self.training:
            recurrent_dropout_mask = self.get_recurrent_dropout_mask(hidden_state)
            hidden_state = GeometricTensor(hidden_state.tensor * recurrent_dropout_mask, hidden_state.type)
        
        gate_conv_input = self._concat_fields([(input, self.in_fields), (hidden_state, self.hidden_fields), (cell_state, self.hidden_fields)])
        gate_conv_output = self.gate_conv(gate_conv_input)
        fz, iz, oz, = self._split_fields(gate_conv_output, self.hidden_fields)
        
        f = self.sigmoid(fz)
        i = self.sigmoid(iz)
        o = self.sigmoid(oz)
        
        update_conv_input = self._concat_fields([(input, self.in_fields), (hidden_state, self.hidden_fields)])
        cell_update_z = self.cell_update_conv(update_conv_input)
        cell_update = self.nonlinearity(cell_update_z)
        
        new_cell_state = GeometricTensor(f.tensor * cell_state.tensor + i.tensor * cell_update.tensor, cell_state.type)
        new_hidden_state = GeometricTensor(o.tensor * self.nonlinearity(new_cell_state).tensor, hidden_state.type)
        
        return new_hidden_state, new_cell_state
    
    def _concat_fields(self, geometric_tensors: tuple[list[GeometricTensor], list[Representation]]) -> GeometricTensor:
        tensors = []
        output_fields = []
        for geom_tensor, fields in geometric_tensors:
            # split height and field dimension
            tensor = geom_tensor.tensor.reshape(-1, self.in_height, sum(field.size for field in fields), *self.in_dims[:2])
            
            tensors.append(tensor)
            output_fields.extend(fields)
            
        output = torch.cat(tensors, dim=2) # concat fields
        
         # combine height and field dimension
        output = output.reshape(-1, self.in_height*sum(field.size for field in output_fields), *self.in_dims[:2])
        
        return GeometricTensor(output, FieldType(self.gspace, self.in_height*output_fields))
        
    def _split_fields(self, geometric_tensor: GeometricTensor, fields: int) -> tuple[GeometricTensor]:
        # split height and field dimension
        batch_size = geometric_tensor.shape[0]
        tensor = geometric_tensor.tensor.reshape(batch_size, self.in_height, -1, *self.in_dims[:2])
            
        channels = sum(field.size for field in fields)
        output_tensors = torch.split(tensor, channels, dim=2)
        
        output_geom_tensors = []
        for output_tensor in output_tensors:
            output_tensor = output_tensor.reshape(batch_size, self.in_height*channels, *self.in_dims[:2])
            output_geom_tensors.append(GeometricTensor(output_tensor, FieldType(self.gspace, self.in_height*fields)))
        
        return output_geom_tensors
    
    def init_state(self, batch_size: int, dims: tuple[int]):
        device = next(self.gate_conv.parameters()).device
        hidden_state = torch.zeros((batch_size, self.hidden_type.size, *dims[:2]), device=device)
        cell_state = torch.zeros((batch_size, self.hidden_type.size, *dims[:2]), device=device)
        
        hidden_state = GeometricTensor(hidden_state, self.hidden_type)
        cell_state = GeometricTensor(cell_state, self.hidden_type)
        
        return hidden_state, cell_state
    
    def get_dropout_mask(self, input):
        if self._dropout_mask is None:
            self._dropout_mask = F.dropout(torch.ones_like(input.tensor), self.drop_rate)
        return self._dropout_mask
    
    def get_recurrent_dropout_mask(self, hidden_state):
        if self._recurrent_dropout_mask is None:
            self._recurrent_dropout_mask = F.dropout(torch.ones_like(hidden_state.tensor), self.drop_rate)
        return self._recurrent_dropout_mask
    
    def reset_dropout_masks(self):
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
        super().__init__()
        
        if type(hidden_fields[0]) is list:
            assert len(hidden_fields) == num_layers
        else:
            hidden_fields = [hidden_fields] * num_layers
            
        self.gspace = gspace
        self.in_fields = in_fields
        self.hidden_fields = hidden_fields
        
        self.in_height = dims[-1]
        self.out_height = self.in_height
        
        self.in_dims = dims
        self.out_dims = dims
        
        self.in_type = FieldType(gspace, self.in_height*self.in_fields)
        self.hidden_types = [FieldType(gspace, self.in_height*hf) for hf in hidden_fields]
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
            
            
    def forward(self, input: list[GeometricTensor], state=None, only_last_output=False):
        """Note: this is not autoregressive (forced encoding)"""
        # shape [batch, inHeight*sum(inFieldsizes), inWidth, inDepth]
        
        batch_size, _, width, depth = input[0].shape
        assert input[0].type == self.in_type
        assert (width, depth) == tuple(self.dims[:2])
        
        if state is None:
            state = self.init_state(batch_size, (self.dims))
            
        for cell in self.cells:
            cell.reset_dropout_masks()
            
        seq_length = len(input)
        for i, cell in enumerate(self.cells):
            layer_hidden_states = []
            for t in range(seq_length):
                hidden_state, cell_state = cell(input[t], state[i])
                state[i] = (hidden_state, cell_state)
                layer_hidden_states.append(hidden_state)
            
            input = layer_hidden_states
        
             
        if only_last_output:
            return [layer_hidden_states[-1]], state
        else:
            return layer_hidden_states, state
    
    def autoregress(self, warmup_input: list[GeometricTensor], steps, state=None, output_whole_warmup=False):
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