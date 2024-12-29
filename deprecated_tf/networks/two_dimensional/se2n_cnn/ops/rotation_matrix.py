"""
Implementation of functions to generate rotation matrices to be applied on tensorflow objects.
Details in MICCAI 2018 paper: "Roto-Translation Covariant Convolutional Networks for Medical Image Analysis".

This module is a modified version of an earlier work by Daniel Worrall under the licence below.

See https://github.com/danielewworrall/harmonicConvolutions for the original implementation.

Modifications include:
- Documentation
- Refactoring
- Tensorflow 2 migration
- Bug fixes
- Unified interface with other models
________________________________________________________________________

Released in June 2018
@author: EJ Bekkers, Eindhoven University of Technology, The Netherlands
@author: MW Lafarge, Eindhoven University of Technology, The Netherlands
________________________________________________________________________

Copyright 2018 Erik J Bekkers and Maxime W Lafarge, Eindhoven University 
of Technology, the Netherlands

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
________________________________________________________________________
"""

import math as m

import numpy as np


def CoordRotationInv(ij: list[int, int], NiNj: list[int, int], theta: float) -> list[int, int]:
    """Appplies the inverse rotation transformation on input coordinates (i,j). The rotation is around the 
    center of the image with dimensions Ni, Nj (resp. # of rows and colums). Input theta is the applied rotation.

    Args:
        ij (list[int, int]): A list of length 2 containing the i and j coordinate as [i,j].
        NiNj (list[int, int]): A list of length 2 containing the dimensions of the 2D domain. 
        theta (float): A real number specifying the angle of rotations.

    Returns:
        list[int, int]:  list of length 2 containing the new coordinate of the inverse rotation, 
        i.e., the old coordinate which was mapped to the new one via (forward) rotation over theta.
    """
    # Define the center of rotation
    centeri = m.floor(NiNj[0] / 2)
    centerj = m.floor(NiNj[1] / 2)

    # Compute the output of the inverse rotation transformation
    ijOld = np.zeros([2])
    ijOld[0] = m.cos(theta) * (ij[0] - centeri) + \
        m.sin(theta) * (ij[1] - centerj) + centeri
    ijOld[1] = -1 * m.sin(theta) * (ij[0] - centeri) + \
        m.cos(theta) * (ij[1] - centerj) + centerj

    # Return the "old" indices
    return ijOld


def LinIntIndicesAndWeights(ij: list[int, int], NiNj: list[int, int]) -> list[list]:
    """Returns, given a target index (i,j), the 4 neighbouring indices and their corresponding weights 
    used for linear interpolation.

    Args:
        ij (list[int, int]): A list of length 2 containing the i and j coordinate as [i,j].
        NiNj (list[int, int]): A list of length 2 containing the dimensions of the 2D domain as [Ni,Nj].

    Returns:
        list[list]: A list index-weight pairs as [[i0,j0,w00],[i0,j1,w01],...].
    """

    # The index where want to obtain the value
    i = ij[0]
    j = ij[1]
    # Image size
    Ni = NiNj[0]
    Nj = NiNj[1]

    # The neighbouring indices
    i1 = int(m.floor(i))  # -- to integer format
    i2 = i1 + 1
    j1 = int(m.floor(j))  # -- to integer format
    j2 = j1 + 1

    # The 1D weights
    ti = i - i1
    tj = j - j1

    # The 2D weights
    w11 = (1 - ti) * (1 - tj)
    w12 = (1 - ti) * tj
    w21 = ti * (1 - tj)
    w22 = ti * tj

    # Only add indices and weights if they fall in the range of the image with
    # dimensions NiNj
    indicesAndWeights = []
    if (0 <= i1 < Ni) and (0 <= j1 < Nj):
        indicesAndWeights.append([i1, j1, w11])
    if (0 <= i1 < Ni) and (0 <= j2 < Nj):
        indicesAndWeights.append([i1, j2, w12])
    if (0 <= i2 < Ni) and (0 <= j1 < Nj):
        indicesAndWeights.append([i2, j1, w21])
    if (0 <= i2 < Ni) and (0 <= j2 < Nj):
        indicesAndWeights.append([i2, j2, w22])

    return indicesAndWeights


def ToLinearIndex(ij: list[int, int], NiNj: list[int, int]) -> int:
    """Returns the linear index of a flattened 2D image that has dimensions [Ni,Nj] before flattening.

    Args:
        ij (list[int, int]): A list of length 2 containing the i and j coordinate as [i,j].
        NiNj (list[int, int]): A list of length 2 containing the dimensions of the 2D domain as [Ni,Nj].

    Returns:
        int: The index in the flattened image.
    """

    return ij[0] * NiNj[0] + ij[1]


def RotationOperatorMatrix(NiNj: list[int, int], theta: float, diskMask: bool = True) -> np.ndarray:
    """Returns the matrix that rotates a square image by R.f, where f is the flattend image 
    (a vector of length Ni*Nj). The resulting vector needs to be repartitioned in to a [Ni,Nj] sized image later. 

    Args:
        NiNj (list[int, int]): A list of length 2 containing the dimensions of the 2D domain as [Ni,Nj].
        theta (float): Real number specifying the rotation angle.
        diskMask (bool, optional): Whether values outside a circular mask are set to zero. Defaults to True.

    Returns:
        np.ndarray: The rotation matrix as a np.array of dimensions [Ni*Nj,Ni*Nj].
    """

    # Image size
    Ni = NiNj[0]
    Nj = NiNj[1]
    cij = m.floor(Ni / 2)  # center

    # Fill the rotation operator matrix
    rotationMatrix = np.zeros([Ni * Nj, Ni * Nj])
    for i in range(0, NiNj[0]):
        for j in range(0, NiNj[0]):
            # Apply a circular mask (disk matrix) if desired
            if not(diskMask) or ((i - cij) * (i - cij) + (j - cij) * (j - cij) <= (cij + 0.5) * (cij + 0.5)):
                # The row index of the operator matrix
                linij = ToLinearIndex([i, j], NiNj)
                # The interpolation points
                ijOld = CoordRotationInv([i, j], NiNj, theta)
                # The indices used for interpolation and their weights
                linIntIndicesAndWeights = LinIntIndicesAndWeights(ijOld, NiNj)
                # Fill the weights in the rotationMatrix
                for indexAndWeight in linIntIndicesAndWeights:
                    indexOld = [indexAndWeight[0], indexAndWeight[1]]
                    linIndexOld = ToLinearIndex(indexOld, NiNj)
                    weight = indexAndWeight[2]
                    rotationMatrix[linij, linIndexOld] = weight
    return rotationMatrix


def MultiRotationOperatorMatrix(NiNj: list[int, int], Ntheta: int, periodicity: float = 2*np.pi, 
                                diskMask: bool = True) -> np.ndarray:
    """Concatenates multiple operator matrices along the first dimension for a direct multi-orientation 
    transformation. 
    I.e., this function returns the matrix that rotates a square image over several angles via R.f, where 
    f is the flattend image (a vector of length Ni*Nj). 
    The dimensions of R are [Ntheta*Ni*Nj,Ni*Nj], with Ntheta the number of orientations sampled from 0 to
    "periodicity". 
    The resulting vector needs to be repartitioned into a [Ntheta,Ni,Nj] stack of rotated images later.

    Args:
        NiNj (list[int, int]): A list of length 2 containing the dimensions of the 2D domain as [Ni,Nj].
        Ntheta (float): An integer specifying the number of rotations.
        periodicity (float, optional): The range of rotations. Defaults to 2*np.pi.
        diskMask (bool, optional): Whether values outside a circular mask are set to zero. Defaults to True.

    Returns:
        np.ndarray: The rotation matrix as a np.array of dimensions [Ntheta*Ni*Nj,Ni*Nj].
    """
    matrices = [None] * Ntheta
    for r in range(Ntheta):
        matrices[r] = RotationOperatorMatrix(
            NiNj,
            periodicity * r / Ntheta,
            diskMask=diskMask)
    return np.concatenate(matrices, axis=0)


def RotationOperatorMatrixSparse(NiNj: list[int, int], theta: float, diskMask: bool = True, 
                                 linIndOffset: int = 0) -> tuple[tuple, tuple]:
    """Returns the idx and vals, where idx is a tuple of 2D indices (also as tuples) and vals the 
    corresponding values. The indices and weights can be converted to a spare tensorflow matrix via 
    R = tf.SparseTensor(idx,vals,[Ni*Nj,Ni*Nj])
    The resulting matrix rotates a square image by R.f, where f is the flattend image (a vector of length Ni*Nj).
    The resulting vector needs to be repartitioned in to a [Ni,Nj] sized image later. 

    Args:
        NiNj (list[int, int]): A list of length 2 containing the dimensions of the 2D domain as [Ni,Nj].
        theta (float): Real number specifying the rotation angle.
        diskMask (bool, optional): Whether values outside a circular mask are set to zero. Defaults to True.
        linIndOffset (int, optional): Offset for the row indices of the rotation matrix. Defaults to 0.

    Returns:
        tuple: A tuple containing the non-zero indices (tuples of length 2).
        tuple: The corresponding values at these indices.
    """
    # Image size
    Ni = NiNj[0]
    Nj = NiNj[1]
    cij = m.floor(Ni / 2)  # center

    # Fill the rotation operator matrix
    idx = [] # This will contain a list of index tuples
    vals = [] # This will contain the corresponding weights
    for i in range(0, NiNj[0]):
        for j in range(0, NiNj[0]):
            # Apply a circular mask (disk matrix) if desired
            if not(diskMask) or ((i - cij) * (i - cij) + (j - cij) * (j - cij) <= (cij + 0.5) * (cij + 0.5)):
                # The row index of the operator matrix
                linij = ToLinearIndex([i, j], NiNj)
                # The interpolation points
                ijOld = CoordRotationInv([i, j], NiNj, theta)
                # The indices used for interpolation and their weights
                linIntIndicesAndWeights = LinIntIndicesAndWeights(ijOld, NiNj)
                # Fill the weights in the rotationMatrix
                for indexAndWeight in linIntIndicesAndWeights:
                    indexOld = [indexAndWeight[0], indexAndWeight[1]]
                    linIndexOld = ToLinearIndex(indexOld, NiNj)
                    weight = indexAndWeight[2]
                    idx = idx + [(linij + linIndOffset, linIndexOld)]
                    vals = vals + [weight]
                    
    # Return the indices and weights as tuples
    return tuple(idx), tuple(vals)


def MultiRotationOperatorMatrixSparse(NiNj: list[int, int], Ntheta: int, periodicity: float = 2*np.pi, 
                                      diskMask: bool = True) -> tuple[tuple, tuple]:
    """Returns the idx and vals, where idx is a tuple of 2D indices (also as tuples) and vals the corresponding values.
    The indices and weights can be converted to a sparse tensorflow matrix via
    R = tf.SparseTensor(idx,vals,[Ntheta*Ni*Nj,Ni*Nj]).
    This matrix rotates a square image over several angles via R.f, where f is the flattend image 
    (a vector of length Ni*Nj).
    The dimensions of R are [Ntheta*Ni*Nj,Ni*Nj], with Ntheta the number of orientations sampled from 0 to "periodicity".
    The resulting vector needs to be repartitioned into a [Ntheta,Ni,Nj] stack of rotated images later.

    Args:
        NiNj (list[int, int]): A list of length 2 containing the dimensions of the 2D domain as [Ni,Nj].
        Ntheta (int): An integer specifying the number of rotations.
        periodicity (float, optional): The range of rotations. Defaults to 2*np.pi.
        diskMask (bool, optional): Whether values outside a circular mask are set to zero. Defaults to True.

    Returns:
        tuple: A tuple containing the non-zero indices (tuples of length 2).
        tuple: The corresponding values at these indices.
    """
    # use linIndOffset to stack across first dimension
    idx = ()
    vals = ()
    for r in range(Ntheta):
        idxr, valsr = RotationOperatorMatrixSparse(
            NiNj, periodicity * r / Ntheta,
            linIndOffset=r * NiNj[0] * NiNj[1],
            diskMask=diskMask)
        idx = idx + idxr
        vals = vals + valsr
    return idx, vals
