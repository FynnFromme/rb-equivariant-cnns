from escnn.gspaces import GSpace3D
from escnn.group import GroupElement

def flipRot2dOnR3(n: int = -1, adjoint: GroupElement = None, maximum_frequency: int = 2) -> GSpace3D:
    r"""

    Describes 2D rotation symmetries along the :math:`Z` axis in the space :math:`\R^3` and a horizontal flip,
    i.e. the rotations inside the
    plane :math:`XY`.
    
    ``adjoint`` is a :class:`~escnn.group.GroupElement` of :class:`~escnn.group.O3`.
    If not ``None`` (which is equivalent to the identity), this specifies another :math:`\SO2` subgroup of :math:`\O3`
    which is adjoint to the :math:`\SO2` subgroup of rotations around the :math:`Z` axis.
    If ``adjoint`` is the group element :math:`A \in \O3`, the new subgroup would then represent rotations around the
    axis :math:`A^{-1} \cdot (0, 0, 1)^T`.

    If ``N > 1``, the gspace models *discrete* rotations by angles which are multiple of :math:`\frac{2\pi}{N}`
    (:class:`~e2cnn.group.CyclicGroup`).
    Otherwise, if ``N=-1``, the gspace models *continuous* planar rotations (:class:`~e2cnn.group.SO2`).
    In that case the parameter ``maximum_frequency`` is required to specify the maximum frequency of the irreps of
    :class:`~e2cnn.group.SO2` (see its documentation for more details)

    Args:
        N (int): number of discrete rotations (integer greater than 1) or ``-1`` for continuous rotations
        adjoint (GroupElement, optional): an element of :math:`\O3`
        maximum_frequency (int): maximum frequency of :class:`~e2cnn.group.SO2`'s irreps if ``N = -1``

    """
    assert isinstance(n, int)
    assert n == -1 or n > 0
    sg_id = False, True, n

    if adjoint is not None:
        sg_id += (adjoint,)
        
    return GSpace3D(sg_id, maximum_frequency=maximum_frequency)