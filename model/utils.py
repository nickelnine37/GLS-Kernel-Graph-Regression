import numpy as np
from numpy import eye as I

from scipy.sparse import csc_matrix, identity as speye
from scipy.linalg import solve

from typing import Union


def vec(X: np.ndarray) -> np.ndarray:
    """
    Equivelant to the mathemmatical vectorisation operation. If X is already a vector, simply
    return it back. Arrays of ndim > 2 raises error.

    Parameters
    ----------
    X   numpy array (N, M)

    Returns
    -------
    x   numpy array (N * M)

    """

    if X.ndim == 1:
        return X
    elif X.ndim == 2:
        return X.T.reshape(-1)
    else:
        raise ValueError


# noinspection PyUnresolvedReferences
def mat(x: np.ndarray, shape: tuple=None, like: np.ndarray=None) -> np.ndarray:
    """
    The reversal of the vectorisation operation. Format a vector into a matrix
    such that x = vec(mat(x, (N, M)))

    Parameters
    ----------
    x       numpy array (N * M)
    shape   tuple - shape to coerse matrix into
    like    numpy array (N, M) - Alternatively pass a 2D numpy array. The vector will
            be coerced into the shape of this array.

    Returns
    -------
    X       numpy array (N, M)

    """

    if shape is None and like is None:
        raise ValueError('Pass either shape or like')

    if shape is not None and like is not None:
        raise ValueError('Pass only one of shape or like')

    if shape is not None:
        if len(shape) != 2:
            raise ValueError(f'shape parameter must be length 2, but it is {shape}')

    else:
        shape = like.shape
        if len(shape) != 2:
            raise ValueError(f'shape of the passed array must be length 2, but it is {shape}')

    if x.ndim == 2:
        if any(s == 1 for s in x.shape):
            return x.reshape(-1).reshape((shape[1], shape[0])).T
        else:
            return x

    elif x.ndim == 1:
        return x.reshape((shape[1], shape[0])).T

    else:
        raise ValueError(f'Cannot vectorise x with {x.ndim} dimensions')


def diag_inds(d: Union[int, list, tuple], N: int) -> (np.ndarray, np.ndarray):
    """
    Reutrn the indices for the d-th diagonal of an NxN matrix

    Parameters
    ----------
    d        Choose the d-th diagonal. 0 -> primary, 1 -> first upper, -1 -> first lower etc.
             If an iterable is passed, return all diagonals listed.
    N        The dimenions of the square matrix

    Returns
    -------
    inds1    numpy array containing vertical indices
    inds2    numpy array containing horizontal indices
    """

    if isinstance(d, int):
        if d > 0:
            return np.arange(0, N - d), np.arange(d, N)
        if d <= 0:
            return np.arange(-d, N), np.arange(0, N + d)

    elif isinstance(d, (list, tuple)):

        inds1, inds2 = zip(*[diag_inds(D, N) for D in d])

        return np.concatenate(inds1), np.concatenate(inds2)


def diag_i(A: np.ndarray, d: int=0) -> np.ndarray:
    """
    Return the d-th diagonal of an NxN array A.
    Parameters
    ----------
    A        NxN numpy array
    d        Choose the d-th diagonal. 0 -> primary, 1 -> first upper, -1 -> first lower etc.
             If an iterable is passed, return all diagonals listed.

    Returns
    -------

    """
    assert A.ndim == 2
    return A[diag_inds(d, A.shape[0])]

def make_B1(T: int, sparse: bool = True) -> Union[np.ndarray, csc_matrix]:
    """
    Create a TxT B1 matrix: That is, a matrix with ones on the first upper and
    lower diagonals.

    Parameters
    ----------
    T           Dimension of resultant matrix: TxT
    sparse      Whether to return a sparse matrix object

    Returns
    -------

    B1          2D Numpy array, or scipy.sparse.csc_matrix

    """

    if sparse:
        M = csc_matrix((np.ones(2 * (T - 1)), diag_inds([-1, 1], T)), shape=(T, T))

    else:
        M = np.zeros((T, T))
        M[diag_inds([-1, 1], T)] = 1

    return M


def make_B2(T: int, sparse: bool = True) -> Union[np.ndarray, csc_matrix]:
    """
    Create a TxT B2 matrix: That is, a the identity matrix with zeros in top left
    and lower right corners.

    Parameters
    ----------
    T           Dimension of resultant matrix: TxT
    sparse      Whether to return a sparse matrix object

    Returns
    -------

    B1          2D Numpy array, or scipy.sparse.csc_matrix
    """

    if sparse:
        M = speye(T, format='csc')

    else:
        M = I(T)

    M[0, 0] = M[-1, -1] = 0
    return M


def make_STi(T: int, theta: float, sparse: bool=True) -> Union[np.ndarray, csc_matrix]:
    """
    Return a TxT inverse autocorrelation matrix, with parameter theta

    Parameters
    ----------
    T           Resultant matrix will be TxT
    theta       Theta parameter
    sparse      Whether the returned array should be sparse

    Returns
    -------
    STi         2D Numpy array, or scipy.sparse.csc_matrix
    """

    if sparse:
        return (speye(T, format='csc') - theta * make_B1(T, sparse) + theta ** 2 * make_B2(T, sparse)) / (1 - theta ** 2)
    else:
        return (np.eye(T) - theta * make_B1(T, sparse) + theta ** 2 * make_B2(T, sparse)) / (1 - theta ** 2)


def make_ST(T: int, theta: float) -> np.ndarray:
    """
    Return a TxT autocorrelation matrix, with parameter theta

    Parameters
    ----------
    T           Resultant matrix will be TxT
    theta       Theta parameter

    Returns
    -------
    ST          2D Numpy array
    """
    return theta ** np.abs(np.arange(T)[:, None] - np.arange(T)[None, :])


def make_chol_ST(T: int, theta: float):
    """
    Return the cholesky decomposition of the TxT  autocorrelation matrix, with parameter theta

    Parameters
    ----------
    T           Resultant matrix will be TxT
    theta       Theta parameter

    Returns
    -------
    ST_chol     2D Numpy array, or scip.sparse array
    """
    M = np.tril(make_ST(T, theta))
    M[:, 1:] *= (1 - theta ** 2) ** 0.5
    return M


def make_chol_STi(T, theta, sparse=True):
    """
    Return the cholesky decomposition of the TxT inverse autocorrelation matrix, with parameter theta

    Parameters
    ----------
    T           Resultant matrix will be TxT
    theta       Theta parameter
    sparse      Whether the returned array should be sparse

    Returns
    -------
    STi_chol    2D Numpy array, or scip.sparse array
    """

    if sparse:
        data = np.concatenate([np.ones(T), -theta * np.ones(T - 1)])
        data[T - 1] = (1 - theta ** 2) ** 0.5
        M = csc_matrix((data, diag_inds([0, -1], T)), shape=(T, T))

    else:
        M = I(T)
        M[-1, -1] = (1 - theta ** 2) ** 0.5
        M[diag_inds(-1, T)] = -theta

    return M / (1 - theta ** 2) ** 0.5


def bsolve(A: np.ndarray, B: np.ndarray, assume_a: str = 'gen'):
    """
    Return the solution X to the problem

    XA = B

    i.e. BA^-1

    ===================  ===============
     generic matrix       assume_a='gen'
     symmetric            assume_a='sym'
     hermitian            assume_a='her'
     positive definite    assume_a='pos'
    ===================  ===============

    Parameters
    ----------
    A
    B
    assume_a

    Returns
    -------

    """
    return solve(A.T, B.T, assume_a=assume_a).T