from torch.nn import Module

def count_trainable_params(model: Module):
    """Returns the number of trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)