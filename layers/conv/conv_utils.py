import math

def conv_output_size(in_size: int, kernel_size: int, stride: int, dilation: int, 
                     pad: bool, equal_pad: bool = False) -> int:
    """Computes the size of a single dimension after applying convolution.

    Args:
        in_size (int): The dimension's size in the input.
        kernel_size (int): The kernel size in that dimension.
        stride (int): The stride in that dimension.
        dilation (int): The dilation in that dimension.
        pad (bool): Whether to use same padding.
        equal_pad (bool, optional): Whether to use the same amount of padding on both sides.
            Defaults to False.

    Returns:
        int: The dimension's size after applying convolution.
    """
    if pad:
        pad_split = required_same_padding(in_size, kernel_size, stride, dilation, split=True)
        padding = 2*pad_split[1] if equal_pad else sum(pad_split)
    else:
        padding = 0

    return ((in_size - dilation*(kernel_size-1) + padding - 1) // stride) + 1


def required_same_padding(in_size: int, kernel_size: int, stride: int, dilation: int,
                          split: bool = False) -> int | tuple[int, int]:
    """Computes for a certain dimension the amount of padding to apply so that the output of 
    a convolution has the same size as the input.

    Args:
        in_size (int): The dimension's size in the input. 
        kernel_size (int): The kernel size in that dimension.
        stride (int): The stride in that dimension.
        dilation (int): The dilation in that dimension.
        split (bool, optional): If `True` the padding is split into a left and right part.
            Otherwise the total amount of padding is returned. Defaults to False.

    Returns:
        int | tuple[int, int]: Either the total amount of required padding or the amount
            of padding required on the left and right seperately.
    """
    out_size = math.ceil(in_size/stride)
    padding = max((out_size-1) * stride - in_size + dilation*(kernel_size-1) + 1, 0)
    
    if split:
        return math.floor(padding/2), math.ceil(padding/2)
    else:
        return padding