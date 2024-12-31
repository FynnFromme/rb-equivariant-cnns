import numpy as np
from torch.nn import Module

from prettytable import PrettyTable
from collections import OrderedDict


def count_trainable_params(model: Module):
    """Returns the number of trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def summary(model: Module, out_shapes: OrderedDict, layer_params: OrderedDict, latent_shape: tuple):
    """Print summary of the model."""
    table = PrettyTable()
    table.field_names = ['Layer', 
                            'Output shape [c, |G|, w, d, h]', 
                            'Parameters']
    table.align['Layer'] = 'l'
    table.align['Output shape [c, |G|, w, d, h]'] = 'r'
    table.align['Parameters'] = 'r'
    
    for layer in out_shapes.keys():
        params = layer_params[layer] if layer in layer_params else 0
        table.add_row([layer, out_shapes[layer], f'{params:,}'])
        
    print(table)
        
    print(f'\nShape of latent space: {latent_shape}')
    
    print(f'\nLatent-Input-Ratio: {np.prod(latent_shape)/np.prod(out_shapes["Input"])*100:.2f}%')

    print(f'\nTrainable parameters: {count_trainable_params(model):,}')