"""
    Chapter 1: Piecewise Polynomial Approximation in 1D

"""
import numpy as np



def MassAssembler1D(x : np.ndarray) -> np.ndarray:
    """
    Routine to assemble the mass matrix for the 1D case.

    Parameters
    ----------
    x : np.ndarray
        The vector (x_0, x_1, ..., x_N) of the nodes coordinates.

    Returns
    -------
    np.ndarray
        The global matrix.
    """

    n = len(x) - 1 # number of subintervals
    M = np.zeros((n+1, n+1)) # initialize the global matrix
    for i in range(0,n): # loop over the subintervals
        h = x[i+1]-x[i] # interval length
        M[i,i] += h/3
        M[i,i+1] += h/6
        M[i+1,i] += h/6
        M[i+1,i+1] += h/3

    return M

def LoadAssembler1D(x : np.ndarray, f : callable) -> np.ndarray:
    """
    Routine to assemble the load vector for the 1D case.

    Parameters
    ----------
    x : np.ndarray
        The vector (x_0, x_1, ..., x_N) of the nodes coordinates.
    f : callable
        The function f(x) to be approximated.

    Returns
    -------
    np.ndarray
        The global vector.
    """

    n = len(x) - 1 # number of subintervals
    b = np.zeros(n+1) # initialize the global vector
    for i in range(0,n): # loop over the subintervals
        h = x[i+1]-x[i] # interval length
        b[i] += (h*f(x[i]))/2
        b[i+1] += (h*f(x[i+1]))/2
    return b

def L2Projector1D(x : np.ndarray, f : callable) -> np.ndarray:
    """
    Routine to compute the L2-projection of a function f(x) on the space of piecewise linear functions.

    Parameters
    ----------
    x : np.ndarray
        The vector (x_0, x_1, ..., x_N) of the nodes coordinates.
    f : callable
        The function f(x) to be approximated.

    Returns
    -------
    np.ndarray
        The global vector.
    """

    M = MassAssembler1D(x) # assemble mass matrix
    b = LoadAssembler1D(x, f) # assemble load vector
    Pf = np.linalg.solve(M, b) # solve the linear system

    return Pf

