from escnn import nn as enn
from escnn import gspaces
from escnn.gspaces import GSpace
from escnn.group import GroupElement
from torch import Tensor

class DataAugmentation:
    def __init__(self, in_height: int, gspace: GSpace = gspaces.flipRot2dOnR2(N=4)):
        self.gspace = gspace
        
        da_irrep_frequencies = (1, 1) if gspace.flips_order > 0 else (1,) 
        self.data_aug_type = enn.FieldType(gspace, 
                                           in_height*[gspace.trivial_repr, 
                                                      gspace.irrep(*da_irrep_frequencies), 
                                                      gspace.trivial_repr])
        
    def __call__(self, *inputs: list[Tensor]) -> Tensor:
        if len(inputs) == 0: return None
        
        transformation = self.gspace.fibergroup.sample()
        
        transformed_inputs = []
        for input in inputs:
            transformed_input = self.transform(input, transformation)
            transformed_inputs.append(transformed_input)
            
        return transformed_inputs[0] if len(inputs) == 1 else transformed_inputs
    
    
    def transform(self, input: Tensor, transformation: GroupElement) -> Tensor:
        input = _to_da_shape(input)
            
        input = enn.GeometricTensor(input, self.data_aug_type)
        transformed_input = input.transform(transformation)
        transformed_input = transformed_input.tensor
        
        transformed_input = _to_original_shape(transformed_input)
        return transformed_input


def _to_da_shape(tensor: Tensor) -> Tensor:
    b, w, d, h, c = tensor.shape
    return tensor.permute(0, 3, 4, 1, 2).reshape(b, h*c, w, d)


def _to_original_shape(tensor: Tensor) -> Tensor:
    b, _, w, d = tensor.shape
    return tensor.reshape(b, -1, 4, w, d).permute(0, 3, 4, 1, 2)