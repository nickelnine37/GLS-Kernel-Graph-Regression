from typing import Union
import numpy as np
from scipy.sparse import identity, csr_matrix


class SelectionMatrix:
    """
    Custom array classes representing binary selection and transposed selection matrices. Implement
    __matmul__ and __rmatmul__  respectively for ease of use, but use efficient slicing rather than
    slower real matrix multiply operations.
    """
    __array_priority__ = 10

    def __init__(self, N: int, n: np.ndarray):
        """
        Initialise a selection matrix

        Parameters
        ----------
        N: int      -> the size of the set we are selecting from
        n: ndarray  -> the indices of the items we want to select
        """
        assert np.max(n) <= N
        self.N = N
        self.N_ = len(n)
        self.n = n

    def __matmul__(self, A: Union[np.ndarray, 'TransposedSelectionMatrix']):
        """
        Select rows from matrix on the right
        """

        if isinstance(A, np.ndarray):
            assert A.shape[0] == self.N, f'Selection Matrix with shape {self.shape} is attempting to hit matrix with shape {A.shape}'
            return A[self.n, :]

        elif isinstance(A, TransposedSelectionMatrix):
            assert A.N == self.N
            assert A.N_ == self.N_
            return identity(self.N_)

    def __rmatmul__(self, A: np.ndarray):
        """
        Should never have to make a left selection as this is not memory efficient
        """
        raise NotImplementedError

    def to_array(self) -> np.ndarray:
        """
        Convert to numpy array
        """
        S = np.zeros((self.N_, self.N))
        S[range(self.N_), self.n] = 1
        return S

    def to_sparse_array(self) -> csr_matrix:
        """
        Convert to sparse array
        """
        return csr_matrix((np.ones(self.N_), (np.arange(self.N_), self.n)), shape=(self.N_, self.N))

    @property
    def T(self):
        """
        Transpose
        """
        return TransposedSelectionMatrix(self.N, self.n)

    def __repr__(self):
        return f"Selection Matrix with shape {self.shape}"

    @property
    def shape(self):
        return (self.N_, self.N)


class TransposedSelectionMatrix:
    __array_priority__ = 10

    def __init__(self, N: int, n: np.ndarray):
        """
        Initialise a transposed selection matrix

        Parameters
        ----------
        N: int      -> the size of the set we are selecting from
        n: ndarray  -> the indices of the items we want to select
        """

        self.N = N
        self.N_ = len(n)
        self.n = n

    def __matmul__(self, A: np.ndarray):
        """
        Should never have to make a right selection as this is not memory efficient
        """
        raise NotImplementedError

    def __rmatmul__(self, A: Union[np.ndarray, 'SelectionMatrix']):
        """
        Select columns from matrix on the left
        """

        if isinstance(A, np.ndarray):
            assert A.shape[1] == self.N, f'Transposed Selection Matrix with shape {self.shape} is being hit by matrix with shape {A.shape}'
            return A[:, self.n]

        elif isinstance(A, TransposedSelectionMatrix):
            assert A.N == self.N
            assert A.N_ == self.N_
            return identity(self.N_)

    def to_array(self) -> np.ndarray:
        """
        Convert to numpy array
        """
        S = np.zeros((self.N, self.N_))
        S[self.n, range(self.N_)] = 1
        return S

    def to_sparse_array(self) -> csr_matrix:
        """
        Convert to sparse array
        """
        return csr_matrix((np.ones(self.N_), (self.n, np.arange(self.N_))), shape=(self.N, self.N_))

    @property
    def T(self):
        """
        Transpose
        """
        return SelectionMatrix(self.N, self.n)

    def __repr__(self):
        return f"Transposed selection Matrix with shape {self.shape}"

    @property
    def shape(self):
        return (self.N, self.N_)


