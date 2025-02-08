import torch

from escnn import nn as enn
from escnn.nn import FieldType, GeometricTensor
from escnn.gspaces import GSpace
from escnn.group import Representation

from layers.conv.steerable_conv import RBSteerableConv
from layers.lstm.steerable_conv_lstm import RBSteerableConvLSTM

from experiments.models import model_utils
from collections import OrderedDict
from typing import Literal

#! training does not work
#! TODO LayerNorm?
#! parameter initialization
class RBSteerableForecaster(enn.EquivariantModule):
    def __init__(
        self,
        gspace: GSpace,
        autoencoder: torch.nn.Module,
        num_layers: int,
        input_channels: int,
        hidden_channels: list[int],
        latent_dims: tuple[int],
        v_kernel_size: int,
        h_kernel_size: int,
        residual_connection: bool = True,
        nonlinearity: Literal['relu', 'elu', 'tanh'] = 'tanh',
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
        
        self.field_type = [gspace.regular_repr]
        self.lstm = RBSteerableConvLSTM(gspace=gspace,
                                        num_layers=num_layers, 
                                        in_fields=input_channels*self.field_type, 
                                        hidden_fields=[hc*self.field_type for hc in hidden_channels],
                                        dims=latent_dims, 
                                        v_kernel_size=v_kernel_size, 
                                        h_kernel_size=h_kernel_size, 
                                        nonlinearity=nonlinearity,
                                        drop_rate=drop_rate,
                                        recurrent_drop_rate=recurrent_drop_rate,
                                        bias=True)
        
        self.dropout = enn.PointwiseDropout(self.lstm.out_type, drop_rate)
        
        self.output_conv = RBSteerableConv(gspace=gspace,
                                           in_fields=hidden_channels[-1]*self.field_type,
                                           out_fields=input_channels*self.field_type,
                                           in_dims=latent_dims,
                                           v_kernel_size=v_kernel_size,
                                           h_kernel_size=h_kernel_size,
                                           v_pad_mode='zero',
                                           h_pad_mode='circular',
                                           bias=True)
        
        self.bias = torch.nn.Parameter(torch.tensor(0.0)) #!
        
        self.in_type, self.out_type = self.lstm.in_type, self.output_conv.out_type
        self.in_fields, self.out_fields = self.lstm.in_fields, self.output_conv.out_fields
        self.in_dims, self.out_dims = self.lstm.in_dims, self.output_conv.out_dims
        self.in_height, self.out_height = self.lstm.in_height, self.output_conv.out_height
        
    def forward(self, warmup_input, steps=1, output_whole_warmup=False):
        # input shape [batch, seq, width, depth, height, channels]
        
        # return warmup_input + self.bias #! 
        
        assert warmup_input.ndim==6, "warmup_input must be a sequence"

        # encode into latent space
        #TODO warmup_latent = self._encode(warmup_input)
        
        #! input b 1 48 48 32 4 -> latent b 1 6 6 4 32
        warmup_latent = torch.repeat_interleave(warmup_input[:, :, :6, :6, :4, :], 8, 5) #!
        
        output_latent = self.forward_latent(warmup_latent, steps, output_whole_warmup)

        # decode into original space
        #TODO output = self._decode(output_latent)
        
        #! latent b 1 6 6 4 32 -> output b 1 48 48 32 4
        output = output_latent[:, :, :, :, :, :4].repeat(1, 1, 8, 8, 8, 1) #!
        
        return output
    
    def forward_latent(self, warmup_input, steps=1, output_whole_warmup=False):
        # input shape (b,warmup_seq,c,w,d,h) -> (b,forecast_seq,c,w,d,h) or (b,warmup_preds+forecast_seq,c,w,d,h)
        # 2 phases: warmup (lstm gets ground truth as input), autoregression: (lstm gets its own outputs as input, for steps > 1)
        # output_whole_warmup: by default only the states after the warmup are outputted. Set to True to output all 
        # len(warmup_input) predictions wuring warmup rather than just the last one
        
        dims = warmup_input.shape[2:5]
        assert tuple(dims) == tuple(self.latent_dims)
        
        warmup_input = self._from_input_shape(warmup_input)
        warmup_length = warmup_input.shape[1]
        #TODO warmup_input = [GeometricTensor(warmup_input[:, t], self.lstm.in_type) for t in range(warmup_length)]
        warmup_input = GeometricTensor(warmup_input[:, -1], self.lstm.in_type) #!
        
        
        lstm_autoregressor = self.lstm.autoregress(warmup_input, steps, output_whole_warmup)
        
        lstm_input = None # warmup_input is already provided to LSTM
        outputs = []
        for i in range(steps):
            lstm_out = lstm_autoregressor.send(lstm_input)
            out = self._apply_output_layer(lstm_out)
            
            #TODO if self.residual_connection:
            #TODO     if i == 0:
            #TODO         res_input = warmup_input if output_whole_warmup else [warmup_input[-1]]
            #TODO     else:
            #TODO         res_input = lstm_input
            #TODO     out = [res + o for res, o in zip(res_input, out)]
            
            #TODO outputs.extend(out)
            outputs.append(out) #!
            #TODO lstm_input = [out[-1]]
            lstm_input = out #!
            
        #TODO output = torch.stack([geom_tensor.tensor for geom_tensor in outputs], dim=1)
        output = out.tensor.unsqueeze(1) #!
        
        output = self._to_output_shape(output)
        return output
    
    def _apply_output_layer(self, lstm_out: list[GeometricTensor]):  
        # shape: (b,h*c,w,d,seq)
        
        hidden_state = lstm_out #!
        return self.output_conv(hidden_state) #!
        
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
        b, s = tensor.shape[:2]
        w, d, h = self.out_dims
        return tensor.reshape(b, s, h, -1, w, d).permute(0, 1, 4, 5, 2, 3)
    
    def summary(self):   
        # LSTM   
        fc_out_shapes = []
        fc_layer_params = []
        for i, cell in enumerate(self.lstm.cells, 1):
            fc_out_shapes.append((f'LSTM{i}', [len(cell.hidden_fields), sum(f.size for f in self.field_type), *cell.dims]))
            fc_layer_params.append((f'LSTM{i}', model_utils.count_trainable_params(cell)))
            
        fc_out_shapes.append(('LSTM-Head', [self.input_channels, sum(f.size for f in self.field_type), *self.latent_dims]))
        fc_layer_params.append((f'LSTM-Head', model_utils.count_trainable_params(self.output_conv)))
        
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
        
    def evaluate_output_shape(self, input_shape: tuple) -> tuple:
        """Compute the shape the output tensor which would be generated by this module when a tensor with shape
        ``input_shape`` is provided as input.
        
        Args:
            input_shape (tuple): shape of the input tensor

        Returns:
            shape of the output tensor
            
        """
        return input_shape