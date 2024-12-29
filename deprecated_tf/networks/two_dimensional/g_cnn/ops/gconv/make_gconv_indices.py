"""
Code for generating indices used in G-convolutions for various groups G.
The indices created by these functions are used to rotate and flip filters on the plane or on a group.
These indices depend only on the filter size, so they are created only once at the beginning of training.
"""

import numpy as np

from ..garray.C4_array import C4
from ..garray.D4_array import D4
from ..garray.p4_array import C4_halfshift
from ..gfunc.p4func_array import P4FuncArray
from ..gfunc.p4mfunc_array import P4MFuncArray
from ..gfunc.z2func_array import Z2FuncArray


def make_c4_z2_indices(ksize: int) -> np.ndarray:
    """Produces an index mapping for all 90° Z2 filter rotations of the form 
    [filter_rot,input_rot,upos,vpos] -> [input_rot',upos',vpos'].
    Size the input is over Z2, input_rot is always 0.

    Args:
        ksize (int): The size of the kernel. Can be both even and odd.

    Returns:
        np.ndarray: The mapping as a np.ndarray of shape [filter_rot,input_rot,upos,vpos,3].
    """
    x = np.random.randn(1, ksize, ksize) # values don't matter
    f = Z2FuncArray(v=x)

    # Apply each 90° Rotation on the filter and get corresponding index mappings
    if ksize % 2 == 0:
        # center is not a point on the grid
        # requires rotation and translation (p4)
        uv = f.left_translation_indices(C4_halfshift[:, None, None, None])
    else:
        # center is a point on the grid
        # only requires rotation (c4)
        uv = f.left_translation_indices(C4[:, None, None, None])
        
    # prepend 0 column for input rotation being always 0
    r = np.zeros(uv.shape[:-1] + (1,))
    ruv = np.c_[r, uv]
    
    return ruv.astype('int32')


def make_c4_p4_indices(ksize: int) -> np.ndarray:
    """Produces an index mapping for all 90° p4 filter rotations of the form 
    [filter_rot,input_rot,upos,vpos] -> [input_rot',upos',vpos'].

    Args:
        ksize (int): The size of the kernel. Can be both even and odd.

    Returns:
        np.ndarray: The mapping as a np.ndarray of shape [filter_rot,input_rot,upos,vpos,3].
    """
    x = np.random.randn(4, ksize, ksize) # values don't matter
    f = P4FuncArray(v=x)

    # Apply each 90° Rotation on the filter and get corresponding index mappings
    if ksize % 2 == 0:
        # center is not a point on the grid
        # requires rotation and translation (p4)
        li = f.left_translation_indices(C4_halfshift[:, None, None, None])
    else:
        # center is a point on the grid
        # only requires rotation (c4)
        li = f.left_translation_indices(C4[:, None, None, None])
    return li.astype('int32')


def make_d4_z2_indices(ksize):
    """Produces an index mapping for all 90° z2 filter rotations and flips of the form 
    [filter_transform,input_transform,upos,vpos] -> [input_transform',upos',vpos'].

    Args:
        ksize (int): The size of the kernel. Must be odd.

    Returns:
        np.ndarray: The mapping as a np.ndarray of shape [filter_transform,input_transform,upos,vpos,3].
    """
    assert ksize % 2 == 1, 'only odd filter sizes supported for p4m' # TODO
    x = np.random.randn(1, ksize, ksize)
    f = Z2FuncArray(v=x)
    uv = f.left_translation_indices(D4.flatten()[:, None, None, None])
    mr = np.zeros(uv.shape[:-1] + (1,))
    mruv = np.c_[mr, uv]
    return mruv.astype('int32')


def make_d4_p4m_indices(ksize):
    """Produces an index mapping for all 90° p4m filter rotations and flips of the form 
    [filter_transform,input_transform,upos,vpos] -> [input_transform',upos',vpos'].

    Args:
        ksize (int): The size of the kernel. Must be odd.

    Returns:
        np.ndarray: The mapping as a np.ndarray of shape [filter_transform,input_transform,upos,vpos,3].
    """
    assert ksize % 2 == 1, 'only odd filter sizes supported for p4m' # TODO
    x = np.random.randn(8, ksize, ksize)
    f = P4MFuncArray(v=x)
    li = f.left_translation_indices(D4.flatten()[:, None, None, None])
    return li.astype('int32')


def flatten_indices(inds: np.ndarray) -> np.ndarray:
    """The Chainer implementation of G-Conv uses indices into a 5D filter tensor (with an additional axis for the
    transformations H. For the tensorflow implementation it was more convenient to flatten the filter tensor into
    a 3D tensor with shape (output channels, input channels, transformations * width * height).

    This function takes indices in the format required for Chainer and turns them into indices into the flat array
    used by tensorflow. 
    Transforms the mapping from [filter_rot,input_rot,upos,vpos] -> [input_rot',upos',vpos'] 
    to [filter_rot,input_rot,upos,vpos] -> index, where index is input_rot' + nti*vpos' + nti*n*upos'. 
    n is the filter size and nti the number of input rotations.

    Args:
        inds (np.ndarray): np.ndarray of shape (output transformations, input transformations, n, n, 3), as output by
            the functions like make_d4_p4m_indices(n).

    Returns:
        np.ndarray: np.ndarray of shape (output transformations, input transformations, n, n)
    """
    n = inds.shape[-2] # filter size
    nti = inds.shape[1] # input transformations
    T = inds[..., 0]  # shape (nto, nti, n, n) # rotation
    U = inds[..., 1]  # shape (nto, nti, n, n) # u-translation
    V = inds[..., 2]  # shape (nto, nti, n, n) # v-translation
    # inds_flat = T * n * n + U * n + V
    inds_flat = U * n * nti + V * nti + T
    return inds_flat