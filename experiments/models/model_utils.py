import numpy as np
from torch.nn import Module

from prettytable import PrettyTable
from collections import OrderedDict


def count_trainable_params(model: Module) -> int:
    """Returns the number of trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def summary(model: Module, out_shapes: OrderedDict, layer_params: OrderedDict, steerable: bool):
    """Print summary of the model."""
    shape = '[w, d, h, c]'
    if steerable: 
        shape += ", |G|"
    
    table = PrettyTable()
    table.field_names = ['Layer', f'Output shape {shape}', 'Parameters']
    table.align['Layer'] = 'l'
    table.align[f'Output shape {shape}'] = 'r'
    table.align['Parameters'] = 'r'
    
    for layer in out_shapes.keys():
        params = layer_params[layer] if layer in layer_params else 0
        table.add_row([layer, out_shapes[layer], f'{params:,}'])
        
    print(table)

    print(f'\nTrainable parameters: {count_trainable_params(model):,}')