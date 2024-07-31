
import numpy as np

from .garray import GArray


class MatrixGArray(GArray):
    """
    Base class for GArrays of groups with matrix parameterization.
    Composition, inversion and the action on vectors is implemented as
    matrix multiplication, matrix inversion and matrix-vector multiplication, respectively.
    """

    def __init__(self, data: np.ndarray, p: str = 'int'):
        """Initializes a MatrixGArray with the data provided.

        Args:
            data (np.ndarray): The array of group elements which parameterization corresponds to p.
            p (str): The parameterization of the given group elements.

        Raises:
            ValueError: If the data does not correspond to the parameterization.
            AssertionError: If the list of parameterizations does not 'mat' or 'hmat'.
        """
        data = np.asarray(data)

        if p == 'int' and data.dtype != int:
            raise ValueError('data.dtype must be int when integer parameterization is used.')

        if 'mat' not in self.parameterizations and 'hmat' not in self.parameterizations:
            raise AssertionError('Subclasses of MatrixGArray should always have a "mat" and/or "hmat" parameterization')

        # add reparameterizations for supported parameterizations
        if 'mat' in self.parameterizations:
            self._reparameterizations[('int', 'mat')] = self.int2mat
            self._reparameterizations[('mat', 'int')] = self.mat2int

        if 'hmat' in self.parameterizations:
            self._reparameterizations[('int', 'hmat')] = self.int2hmat
            self._reparameterizations[('hmat', 'int')] = self.hmat2int

        if 'mat' in self.parameterizations and 'hmat' in self.parameterizations:
            self._reparameterizations[('hmat', 'mat')] = self.hmat2mat
            self._reparameterizations[('mat', 'hmat')] = self.mat2hmat

        super(MatrixGArray, self).__init__(data, p)

    def inv(self) -> 'MatrixGArray':
        """Compute the inverse of the group elements.

        Returns:
            MatrixGArray: MatrixGArray of the same shape as self, containing inverses of each element in self.
        """
        # perform inversion as matrix inverion in either mat or hmat parameterization 
        mat_p = 'mat' if 'mat' in self.parameterizations else 'hmat'
        self_mat = self.reparameterize(mat_p).data
        self_mat_inv = np.linalg.inv(self_mat)
        self_mat_inv = np.round(self_mat_inv, 0).astype(self_mat.dtype)
        
        # transform back to the original parameterization
        return self.factory(data=self_mat_inv, p=mat_p).reparameterize(self.p)

    def left_action_mat(self, other: 'GArray') -> 'GArray':
        """Performs the left action in the matrix parameterization.
        Requires both self and other to support this parameterization.

        Args:
            other (GArray): The GArray the group elements acts on.

        Returns:
            GArray: The resulting GArray of other's type and other's parameterization.
        """
        self_mat = self.reparameterize('mat').data
        other_mat = other.reparameterize('mat').data
        c_mat = np.einsum('...ij,...jk->...ik', self_mat, other_mat) # matrix multiplication of group elements
        return other.factory(data=c_mat, p='mat').reparameterize(other.p)

    def left_action_hmat(self, other: 'GArray') -> 'GArray':
        """Performs the left action in the homogeneous matrix (hmat) parameterization.
        Requires both self and other to support this parameterization.

        Args:
            other (GArray): The GArray the group elements acts on.

        Returns:
            GArray: The resulting GArray of other's type and other's parameterization.
        """
        self_hmat = self.reparameterize('hmat').data
        other_hmat = other.reparameterize('hmat').data
        c_hmat = np.einsum('...ij,...jk->...ik', self_hmat, other_hmat) # matrix multiplication of group elements
        return other.factory(data=c_hmat, p='hmat').reparameterize(other.p)

    def left_action_vec(self, other: 'GArray') -> 'GArray':
        """Performs the left action by matrix-vector multiplication.
        Requires self to support matrix parameterization and other to be in int parameterization.

        Args:
            other (GArray): The GArray the group elements acts on.

        Returns:
            GArray: The resulting GArray of other's type and other's parameterization.
            
        Raises:
            AssertionError: If other is not in int parameterization.
        """
        self_mat = self.reparameterize('mat').data
        assert other.p == 'int'  # TODO
        out = np.einsum('...ij,...j->...i', self_mat, other.data) # matrix-vector multiplication of group elements
        return other.factory(data=out, p=other.p)

    def left_action_hvec(self, other: 'GArray') -> 'GArray':
        """Performs the left action by hmat-vector multiplication.
        Requires self to support hmat parameterization and other to be in int parameterization.

        Args:
            other (GArray): The GArray the group elements acts on.

        Returns:
            GArray: The resulting GArray of other's type and other's parameterization.
            
        Raises:
            AssertionError: If other is not in int parameterization.
        """
        self_hmat = self.reparameterize('hmat').data
        assert other.p == 'int'  # TODO
        self_mat = self_hmat[..., :-1, :-1]
        out = np.einsum('...ij,...j->...i', self_mat, other.data) + self_hmat[..., :-1, -1]
        return other.factory(data=out, p=other.p)

    def int2mat(self, int_data: np.ndarray) -> np.ndarray:
        """Implementation is group specific."""
        raise NotImplementedError()

    def mat2int(self, mat_data: np.ndarray) -> np.ndarray:
        """Implementation is group specific."""
        raise NotImplementedError()

    def mat2hmat(self, mat_data: np.ndarray) -> np.ndarray:
        """Transforms matrix parameterized group elements into homogeneous matrix parameterization 
        by extending them with zeros into both dimensions.

        Args:
            mat_data (np.ndarray): The data with group elements in matrix parameterization.

        Returns:
            np.ndarray: The data with reparameterized group elements.
        """
        n, m = self._g_shapes['mat']
        out = np.zeros(mat_data.shape[:-2] + (n + 1, m + 1), dtype=mat_data.dtype)
        out[..., :n, :m] = mat_data
        return out

    def hmat2mat(self, hmat_data: np.ndarray) -> np.ndarray:
        """Transforms homogeneous matrix parameterized group elements into matrix parameterization 
        by removing one row and column.

        Args:
            mat_data (np.ndarray): The data with group elements in hmat parameterization.

        Returns:
            np.ndarray: The data with reparameterized group elements.
        """
        return hmat_data[..., :-1, :-1]

    def int2hmat(self, int_data: np.ndarray) -> np.ndarray:
        """Transforms int parameterized group elements into homogeneous matrix parameterization.

        Args:
            mat_data (np.ndarray): The data with group elements in int parameterization.

        Returns:
            np.ndarray: The data with reparameterized group elements.
        """
        # The exact behaviour can either be defined by implementing int2mat in a subclass or overriding this method instead
        return self.mat2hmat(self.int2mat(int_data))

    def hmat2int(self, hmat_data: np.ndarray) -> np.ndarray:
        """Transforms homogeneous matrix parameterized group elements into int parameterization.

        Args:
            mat_data (np.ndarray): The data with group elements in homogeneous matrix parameterization.

        Returns:
            np.ndarray: The data with reparameterized group elements.
        """
        # The exact behaviour can either be defined by implementing int2mat in a subclass or overriding this method instead
        return self.mat2int(self.hmat2mat(hmat_data))
