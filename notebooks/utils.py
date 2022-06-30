import numpy as np
from typing import Callable


def matrix_derivative_numerical(f: Callable[[np.ndarray], float], X: np.ndarray):
    """
    Find a numerical approximation to the derivative of a scalar function of a matrix
    at a point X

    Parameters
    ----------
    f           A scalar function of a matrix of shape (T, M)
    X           The place to evaluate the derivative (T, M)

    Returns
    -------
    df          The approximate derivative (T, M)
    """

    out = np.zeros_like(X)
    dx = 0.001

    T, M = X.shape

    for i in range(T):
        for j in range(M):
            W_ = X.copy()
            _W = X.copy()
            W_[i, j] += dx / 2
            _W[i, j] -= dx / 2
            out[i, j] = (f(W_) - f(_W)) / dx

    return out


def vector_derivative_numerical(f: Callable[[np.ndarray], float], x: np.ndarray):
    """
    Find a numerical approximation to the derivative of a scalar function of a vector
    at a point x

    Parameters
    ----------
    f           A scalar function of a vector of shape (N, )
    x           The place to evaluate the derivative (N, )

    Returns
    -------
    df          The approximate derivative (N, )
    """

    out = np.zeros_like(x)

    dx = 0.001

    for i in range(len(x)):
        x_ = x.copy()
        _x = x.copy()
        x_[i] += dx / 2
        _x[i] -= dx / 2
        out[i] = (f(x_) - f(_x)) / dx

    return out


def hessian_numerical(f: Callable[[np.ndarray], float], x: np.ndarray):
    """
    Find the approximate Hessian for a scalar function f on a vector at a point x

    Parameters
    ----------
    f           A scalar function of a vector of shape (N, )
    x           The place to evaluate the Hessian (N, )

    Returns
    -------
    H           The approximate Hessian (N, N)
    """

    dx = 0.001
    N = len(x)
    out = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            def deriv(g, x, k):
                x_ = x.copy()
                _x = x.copy()
                x_[k] += dx / 2
                _x[k] -= dx / 2
                return (g(x_) - g(_x)) / dx

            out[i, j] = deriv(lambda y: deriv(f, y, i), x, j)

    return out
