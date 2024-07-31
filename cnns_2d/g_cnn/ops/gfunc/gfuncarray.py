
import copy

import numpy as np

from ..garray.garray import GArray


class GFuncArray(object):

    def __init__(self, v: np.ndarray, i2g: GArray):
        """
        A GFunc is a discretely sampled function on a group or homogeneous space G.
        The GFuncArray stores an array of GFuncs,
        together with a map from G to an index set I (the set of sampling points) and the inverse of this map.

        The ndarray v can be thought of as a map
         v : J x I -> R
        from an index set J x I to real numbers.
        The index set J may have arbitrary shape, and each index in j identifies a GFunc.
        The index set I is the set of valid indices to the ndarray v.
        From here on, consider a single GFunc v : I -> R

        The GArray i2g can be thought of as a map
          i2g: I -> G
        that takes indices from I and produces a group element g in G.

        The map i2g is required to be invertible, and its inverse
         g2i : G -> I
        is implemented in the function g2i of a subclass.

        So we have the following diagram:
              i2g
          I <-----> G
          |   g2i
        v |
          |
          V
          R

        So v implicitly defines a function v' on G:
        v'(g) = v(g2i(g))

        If we have a map T: G - > G (e.g. left multiplication by g^-1), that we want to precompose with v',
         w'(g) = v'(T(g))

        we can get the corresponding map v by composing maps like this:
        I ---> G ---> G ---> I ---> R
          i2g     T     g2i     v
        to obtain the transformed function w : I -> R.
        This class knows how to produce such a w as an ndarray that directly maps indices to numbers,
        (and such that the indices correspond to group elements by the same maps i2g and g2i)

        Args:
            v (np.ndarray): A numpy.ndarray of values corresponding to the sample points.
            i2g (GArray): A GArray of sample points. The sample points are elements of G or H. 
                Each of these sample points has a distinct value for each GFunc in v.

        Raises:
            TypeError: If the arguments are of a different type than expected.
            ValueError: If the shape of I does not match the shape of i2g.
        """

        if not isinstance(i2g, GArray):
            raise TypeError('i2g must be of type GArray, got' + str(type(i2g)) + ' instead.')

        if not isinstance(v, np.ndarray):
            raise TypeError('v must be of type np.ndarray, got ' + str(type(v)) + ' instead.')

        if i2g.shape != v.shape[-i2g.ndim:]:  # TODO: allow vector-valued gfunc, or leave this to Section?
            raise ValueError('The trailing axes of v must match the shape of i2g. Got ' +
                             str(i2g.shape) + ' and ' + str(v.shape) + '.')

        self.i2g = i2g
        self.v = v

    def __call__(self, sample_points: 'GArray') -> 'GFuncArray':
        """Evaluate each G-function at the sample points. Maps a GArray to a GFuncArray with f.v holding 
        the function values of the sample points for all G-functions in the GFuncArray: [function, sample_point] -> R

        Args:
            sample_points (GArray): GArray containing the group elements to retrieve the function 
                values for. Must be the same type i2g.

        Raises:
            TypeError: If the given GArray is not of the same type as i2g.

        Returns:
            GFuncArray: A function whose v contains the values of the different G-functions at all sample points.
        """
        if not isinstance(sample_points, type(self.i2g)):
            raise TypeError('Invalid type ' + str(type(sample_points)) + ' expected ' + str(type(self.i2g)))

        si = self.g2i(sample_points) # indices of the corresponding group elements
        
        # retrieve function values of sample indices
        inds = [Ellipsis] + [si[..., i] for i in range(si.shape[-1])]
        vi = self.v[*inds]
        
        # create new function with values
        ret = copy.copy(self)
        ret.v = vi # [function, sample_point] -> R
        return ret

    def __getitem__(self, item) -> 'GFuncArray':
        """Get an element from the array of G-funcs."""
        # TODO bounds / dim checking
        ret = copy.copy(self)
        ret.v = self.v[item]
        return ret

    def __mul__(self, other: GArray) -> 'GFuncArray':
        # Compute self * other
        if isinstance(other, GArray):
            gp = self.right_translation_points(other)
            return self(gp)
        else:
            # Python assumes we *return* NotImplemented instead of raising NotImplementedError,
            # when we dont know how to left multiply the given type of object by self.
            return NotImplemented

        # Compute other * self
    def __rmul__(self, other: GArray) -> 'GFuncArray':
        """Evaluate the function for all transformations of the pooling domain.

        Args:
            other (GArray): The GArray containing the transformations.

        Returns:
            GFuncArray: with v.shape=[GFuncArrayShape,GArrayShape,i2gShape]
        """
        if isinstance(other, GArray):
            gp = self.left_translation_points(other)
            return self(gp)
        else:
            # Python assumes we *return* NotImplemented instead of raising NotImplementedError,
            # when we dont know how to left multiply the given type of object by self.
            return NotImplemented

    def g2i(self, g: GArray) -> np.ndarray:
        """Implementation is group specific."""
        raise NotImplementedError()

    def left_translation_points(self, g: GArray) -> GArray:
        """Transforms Function Domain inversly by transformations out of g.

        Args:
            g (GArray): GArray of transformations of shape [GArrayShape, i2g_dim * newaxis]

        Returns:
            GArray: with GArray.shape=[GArrayShape, i2gShape]
        """
        return g.inv() * self.i2g

    def right_translation_points(self, g: GArray) -> GArray:
        return self.i2g * g

    def left_translation_indices(self, g: GArray) -> np.ndarray:
        """Transforms Function Domain inversly by transformations out of g. The resulting 
        domain in form of group elements is then transformed into indices.

        Args:
            g (GArray):  GArray of transformations of shape [GArrayShape, i2g_dim * newaxis]

        Returns:
            np.ndarray: of shape [GArrayShape, i2gShape, GroupElementShape]
        """
        ginv_s = self.left_translation_points(g)
        ginv_s_inds = self.g2i(ginv_s)
        return ginv_s_inds

    def right_translation_indices(self, g: GArray) -> np.ndarray:
        sg = self.right_translation_points(g)
        sg_inds = self.g2i(sg)
        return sg_inds

    @property
    def ndim(self) -> int:
        """Calculates the dimension of function indexing.

        Returns:
            int: The dimension of function indexing.
        """
        return self.v.ndim - self.i2g.ndim

    @property
    def shape(self) -> list:
        """Calculates the shape of function indexing.

        Returns:
            int: The shape of function indexing.
        """
        return self.v.shape[:self.ndim]

    @property
    def f_shape(self) -> list:
        """Calculates the shape of the function domain.

        Returns:
            int: The shape of the function domain.
        """
        return self.i2g.shape

    @property
    def f_ndim(self) -> int:
        """Calculates the dimension of the function domain.

        Returns:
            int: The dimension of the function domain.
        """
        return self.i2g.ndim
