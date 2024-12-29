import numpy as np

from ..garray import p4_array as p4a
from .gfuncarray import GFuncArray


class P4FuncArray(GFuncArray):

    def __init__(self, v: np.ndarray, umin: int = None, umax: int = None, vmin: int = None, vmax: int = None):
        """P4 function array with 4 rotation channels (planar patches) each of size (umin+umax+1) x (vmin+vmax+1).

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
            umax = hnu - (nu % 2 == 0)
            vmin = -hnv
            vmax = hnv - (nv % 2 == 0)

        self.umin = umin
        self.umax = umax
        self.vmin = vmin
        self.vmax = vmax

        # four rotation channels centered at origin
        i2g = p4a.meshgrid(
            r=p4a.r_range(0, 4),
            u=p4a.u_range(self.umin, self.umax + 1),
            v=p4a.v_range(self.vmin, self.vmax + 1)
        )

        super(P4FuncArray, self).__init__(v=v, i2g=i2g)

    def g2i(self, g: p4a.P4Array) -> np.ndarray:
        """Transforms an array of p4 elements to indices that can be used to access v.

        Args:
            g (p4_array.P4Array): Array of p4 elements.

        Returns:
            np.ndarray: The corresponding indices.
        """
        # TODO: check validity of indices and wrap / clamp if necessary
        # (or do this in a separate function, so that this function can be more easily tested?)

        gint = g.reparameterize('int').data.copy()
        gint[..., 1] -= self.umin
        gint[..., 2] -= self.vmin
        return gint


def tst():
    from garray.p4_array import (P4Array, meshgrid, rotation, translation,
                                 u_range, v_range)

    x = np.random.randn(4, 3, 3)
    c = meshgrid(u=u_range(-1, 2), v=v_range(-1, 2))

    f = P4FuncArray(v=x)

    g = rotation(1, center=(0, 0))
    li = f.left_translation_indices(g)
    lp = f.left_translation_points(g)

    # gfi = f[li]
    gfp = f(lp)
    gf = g * f
    gfi = f.v[li[..., 0], li[..., 1], li[..., 2]]

    return x, c, f, li, gf, gfp, gfi