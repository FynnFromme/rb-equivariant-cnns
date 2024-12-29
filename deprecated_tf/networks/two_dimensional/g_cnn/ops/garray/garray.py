
import copy

import numpy as np

# TODO: add checks in constructor to make sure data argument is well formed (for the given parameterization).
# TODO: for example, for a finite group, when p=='int', we want data >= 0 and data <= order_of_G


class GArray(object):
    """
    GArray is a wrapper of numpy.ndarray that can store group elements instead of numbers.
    Subclasses of GArray implement the needed functionality for specific groups G.

    A GArray has a shape (how many group elements are in the array),
    and a g_shape, which is the shape used to store group element itself (e.g. (3, 3) for a 3x3 matrix).
    The user of a GArray usually doesn't need to know the g_shape, or even the group G.
    GArrays should be fully gufunc compatible; i.e. they support broadcasting according to the rules of numpy.
    A GArray of a given shape broadcasts just like a numpy array of that shape, regardless of the g_shape.

    A group may have multiple parameterizations, each with its own g_shape.
    Group elements can be composed and compared (using the * and == operators) irrespective of their parameterization.
    """

    # To be set in subclass
    parameterizations = [] # list of paramaterizations supported
    _g_shapes = {} # parameterization -> shape of group elements
    _left_actions = {} # type of element it acts on -> function
    _reparameterizations = {} # (p_from, p_to) -> function
    _group_name = 'GArray Base Class'

    def __init__(self, data: np.ndarray, p: str):
        """Initializes a GArray with the data provided.

        Args:
            data (np.ndarray): The array of group elements which parameterization corresponds to p.
            p (str): The parameterization of the given group elements.

        Raises:
            TypeError: If the data is not provided as a np.ndarray.
            ValueError: If the parameterization p is not know.
            ValueError: If the shape of the group elements does not correspond to the given parameterization.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError('data should be of type np.ndarray, got ' + str(type(data)) + ' instead.')

        if p not in self.parameterizations:
            raise ValueError('Unknown parameterization: ' + str(p))

        self.data = data
        self.p = p
        self.g_shape = self._g_shapes[p]
        self.shape = data.shape[:data.ndim - self.g_ndim]

        if self.data.shape[self.ndim:] != self.g_shape:
            raise ValueError('Invalid data shape. Expected shape ' + str(self.g_shape) +
                             ' for parameterization ' + str(p) +
                             '. Got data shape ' + str(self.data.shape[self.ndim:]) + ' instead.')

    def inv(self) -> 'GArray':
        """Compute the inverse of the group elements.

        Returns:
            GArray: GArray of the same shape as self, containing inverses of each element in self.
        """
        raise NotImplementedError()

    def reparameterize(self, p: str) -> 'GArray':
        """Return a GArray containing the same group elements in the requested parameterization p.
        If p is the same as the current parameterization, this function returns self.

        Args:
            p (str): The requested parameterization. Must be an element of self.parameterizations.

        Raises:
            ValueError: If the parameterization is unknown or the parameterization is not implemented for the parameterization.

        Returns:
            GArray: GArray subclass with reparameterized elements.
        """
        if p == self.p:
            # nothing to do
            return self

        if p not in self.parameterizations:
            raise ValueError('Unknown parameterization:' + str(p))

        if not (self.p, p) in self._reparameterizations:
            return ValueError('No reparameterization implemented for ' + self.p + ' -> ' + str(p))

        new_data = self._reparameterizations[(self.p, p)](self.data)
        return self.factory(data=new_data, p=p)

    def reshape(self, *shape) -> 'GArray':
        """Creates a new instance with the reshaped data without changing the shape of the group elements.

        Args:
            shape (tuple[int]): The new shape as a single tuple or multiple arguments.
        
        Returns:
            GArray: New GArray with reshaped data.
        """
        shape = shape[0] if isinstance(shape[0], tuple) else shape
        full_shape = shape + self.g_shape
        new = copy.copy(self)
        new.data = self.data.reshape(full_shape)
        new.shape = shape
        return new

    def flatten(self) -> 'GArray':
        """Creates a new instance with the data flattened to a single dimension without 
        changing the shape of the group elements.

        Returns:
            GArray: The flattened GArray.
        """
        return self.reshape(np.prod(self.shape))

    def __mul__(self, other: 'GArray') -> 'GArray':
        """Act on another GArray from the left (self*other).

        If the arrays do not have the same shape for the loop dimensions, they are broadcast together.

        The left action is chosen from self.left_actions depending on the type of other;
        this way, a GArray subclass can act on various other compatible GArray subclasses.

        This function will still work if self and other have a different parameterization.
        The output is always returned in the other's parameterization.

        Args:
            other (GArray): The GArray self acts on.

        Returns:
            GArray: The resulting GArray of other's type and other's parameterization.
        """
        for garray_type in self._left_actions:
            if isinstance(other, garray_type):
                return self._left_actions[garray_type](self, other)
        return NotImplemented

    def __eq__(self, other: 'GArray') -> np.ndarray:
        """Elementwise equality test of GArrays.
        Group elements are considered equal if, after reparameterization, they are numerically identical.

        Args:
            other (GArray): GArray to be compared to.

        Returns:
            np.ndarray: A boolean numpy.ndarray of shape self.shape.
        """
        
        if isinstance(other, self.__class__) or isinstance(self, other.__class__):
            #! only collapses 1 dimension and not the dimensions of group elements
            return (self.data == other.reparameterize(self.p).data).all(axis=-1) 
        else:
            return NotImplemented

    def __ne__(self, other: 'GArray') -> np.ndarray:
        """Elementwise inequality test of GArrays.
        Group elements are considered equal if, after reparameterization, they are numerically identical.

        Args:
            other (GArray): GArray to be compared to.

        Returns:
            np.ndarray: A boolean numpy.ndarray of shape self.shape.
        """
        if isinstance(other, self.__class__) or isinstance(self, other.__class__):
            #! only collapses 1 dimension and not the dimensions of group elements
            return (self.data != other.reparameterize(self.p).data).any(axis=-1)
        else:
            return NotImplemented

    def __len__(self) -> int:
        """Returns the size of the first dimension.

        Returns:
            int: The size of the first dimension.
        """
        if len(self.shape) > 0:
            return self.shape[0]
        else:
            return 1
    
    def __getitem__(self, key: tuple[int]) -> 'GArray':
        """Supports accessing elements and subdimensional arrays as np.ndarrays do."""
        # We return a factory here instead of self.__class__(..) so that a subclass
        # can decide what type the result should have.
        # For instance, a FiniteGroup may wish to return an instance of a different GArray instead of a FiniteGroup.
        return self.factory(data=self.data[key], p=self.p)

    # def __setitem__(self, key, value):
    #    raise NotImplementedError()  # TODO

    def __delitem__(self, key: tuple[int]) -> None:
        """Raise an error to mimic the behaviour of np.ndarray."""
        
        raise ValueError('cannot delete garray elements')

    def __iter__(self):
        """Iterate over the first dimension of the array."""
        for i in range(self.shape[0]):
            yield self[i]

    def __contains__(self, item):
        return (self == item).any()

    
    def factory(self, *args, **kwargs) -> 'GArray':
        """Creates a new GArray of the given arguments.
        Factory is used to create new instances from a given instance, e.g. when using __getitem__ or inv()
        In some cases (e.g. FiniteGroup), we may wish to instantiate a superclass instead of self.__class__
        Example: D4Group instantiates a D4Array when an element is selected.

        Returns:
            GArray: The initialted GArray.
        """
        return self.__class__(*args, **kwargs)

    @property
    def size(self) -> int:
        """Computes the number of group elements in the GArray.

        Returns:
            int: The number of group elements in the GArray.
        """
        # Usually, np.prod(self.shape) returns an int because self.shape contains ints.
        # However, if self.shape == (), np.prod(self.shape) returns the float 1.0,
        # so we convert to int.
        return int(np.prod(self.shape))

    @property
    def g_ndim(self) -> int:
        """Returns the number of dimensions of each group element in this GArray, 
        for the current parameterization.

        Returns:
            int: The dimensions of each group element in this GArray, 
                for the current parameterization.
        """
        return len(self.g_shape)

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the GArray, where the dimensions 
        within the group elements are ignored.

        Returns:
            int: Returns the number of dimensions of the GArray, where the dimensions 
                within the group elements are ignored.
        """
        return len(self.shape)

    def __repr__(self) -> str:
        return self._group_name + self.data.__repr__()