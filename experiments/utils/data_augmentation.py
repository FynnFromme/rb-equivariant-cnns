from escnn import nn as enn
from escnn import gspaces
from escnn.gspaces import GSpace
from torch import Tensor

class DataAugmentation:
    def __init__(self, in_height: int, gspace: GSpace = gspaces.flipRot2dOnR2(N=4)):
        self.gspace = gspace
        da_irrep_frequencies = (1, 1) if gspace.flips_order > 0 else (1,) # depending whether using Cn or Dn group  
        self.data_aug_type = enn.FieldType(gspace, 
                                           in_height*[gspace.trivial_repr, 
                                                      gspace.irrep(*da_irrep_frequencies), 
                                                      gspace.trivial_repr])
        
    def __call__(self, *inputs: list[Tensor]) -> Tensor:
        if len(inputs) == 0: return None
        
        transformation = self.gspace.fibergroup.sample()
        
        transformed_inputs = []
        for input in inputs:
            input = enn.GeometricTensor(input, self.data_aug_type)
            transformed_input = input.transform(transformation)
            transformed_inputs.append(transformed_input.tensor)
            
        return transformed_inputs[0] if len(inputs) == 1 else transformed_inputs