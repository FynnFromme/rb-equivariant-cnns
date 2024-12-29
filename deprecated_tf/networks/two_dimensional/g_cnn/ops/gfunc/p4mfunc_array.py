import numpy as np

from ..garray import p4m_array as p4ma
from .gfuncarray import GFuncArray


class P4MFuncArray(GFuncArray):

    def __init__(self, v: np.ndarray, umin: int = None, umax: int = None, vmin: int = None, vmax: int = None):
        """P4M function array with 8 rotation(-flip) channels (planar patches) each of size (umin+umax+1) x (vmin+vmax+1).

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

            # If (u, v) ranges are not given, determine them from the shape of v, assuming the grid is centered.
            nu, nv = v.shape[-2:]

            hnu = nu // 2
            hnv = nv // 2

            umin = -hnu
            umax = hnu
            vmin = -hnv
            vmax = hnv

        self.umin = umin
        self.umax = umax
        self.vmin = vmin
        self.vmax = vmax

        i2g = p4ma.meshgrid(
            m=p4ma.m_range(),
            r=p4ma.r_range(0, 4),
            u=p4ma.u_range(self.umin, self.umax + 1),
            v=p4ma.v_range(self.vmin, self.vmax + 1)
        )

        if v.shape[-3] == 8:
            i2g = i2g.reshape(8, i2g.shape[-2], i2g.shape[-1])
            self.flat_stabilizer = True
        else:
            self.flat_stabilizer = False

        super(P4MFuncArray, self).__init__(v=v, i2g=i2g)

    def g2i(self, g: p4ma.P4MArray) -> np.ndarray:
        """Transforms an array of p4m elements to indices that can be used to access v.

        Args:
            g (p4m_array.P4MArray): Array of p4m elements.

        Returns:
            np.ndarray: The corresponding indices.
        """
        # TODO: check validity of indices and wrap / clamp if necessary
        # (or do this in a separate function, so that this function can be more easily tested?)

        gint = g.reparameterize('int').data.copy()
        gint[..., 2] -= self.umin
        gint[..., 3] -= self.vmin

        if self.flat_stabilizer:
            gint[..., 1] += gint[..., 0] * 4
            gint = gint[..., 1:]

        return gint
