import numpy as np

from ..garray import Z2_array as z2a
from .gfuncarray import GFuncArray


class Z2FuncArray(GFuncArray):

    def __init__(self, v: np.ndarray, umin: int = None, umax: int = None, vmin: int = None, vmax: int = None):
        """Z2 function array with 1 planar patche of size (umin+umax+1) x (vmin+vmax+1).

        Args:
            v (np.ndarray): The function values of the shape [FuncArrayShape,4,umin+umax+1,vmin+vmax+1]
            umin (int, optional): The minimum u value. Derrived from v.shape by default.
            umax (int, optional): The maximum v value. Derrived from v.shape by default.
            vmin (int, optional): The minimum u value. Derrived from v.shape by default.
            vmax (int, optional): The maximum v value. Derrived from v.shape by default.

        Raises:
            ValueError: If only some of the umin,umax,vmin,vmax values are given.
        """
        if umin is None or umax is None or vmin is None or vmax is None:
            if not (umin is None and umax is None and vmin is None and vmax is None):
                raise ValueError('Either all or none of umin, umax, vmin, vmax must equal None')

            # If (u, v) ranges are not given, determine them from the shape of v,
            # assuming the grid is centered.
            nu, nv = v.shape[-2:]

            hnu = nu // 2
            hnv = nv // 2

            umin = -hnu
            umax = hnu - (nu % 2 == 0)
            vmin = -hnv
            vmax = hnv - (nv % 2 == 0)

        self.umin = umin
        self.umax = umax
        self.vmin = vmin
        self.vmax = vmax

        i2g = z2a.meshgrid(
            u=z2a.u_range(self.umin, self.umax + 1),
            v=z2a.v_range(self.vmin, self.vmax + 1)
        )

        super(Z2FuncArray, self).__init__(v=v, i2g=i2g)

    def g2i(self, g: z2a.Z2Array) -> np.ndarray:
        """Transforms an array of Z2 elements to indices that can be used to access v.

        Args:
            g (Z2_array.Z2Array): Array of Z2 elements.

        Returns:
            np.ndarray: The corresponding indices.
        """
        # TODO: check validity of indices and wrap / clamp if necessary
        # (or do this in a separate function, so that this function can be more easily tested?)

        gint = g.reparameterize('int').data.copy()
        gint[..., 0] -= self.umin
        gint[..., 1] -= self.vmin
        return gint
