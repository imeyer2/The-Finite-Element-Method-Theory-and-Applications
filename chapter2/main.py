"""
    Chapter 2: The Finite Element Method in 1D

    Assuming the model problem is

    -(au')' = f  x in [0,L]
    au'(0) = k_0(u(0) - g_0)
    -au'(L) = k_1(u(L) - g_L)

    where , k_0, k_l >= 0, g_0, g_L are constants, and f is a given function and a>0 is a given function.
"""

import numpy as np
from chapter1.main import LoadAssembler1D

def StiffnessAssembler1D(x : np.ndarray, a : callable, kappa_0 : float, kappa_L : float):
    """
    Routine to assemble the stiffness matrix for the 1D case.
    
    Parameters
    ----------
    x : np.ndarray
        The vector (x_0, x_1, ..., x_N) of the nodes coordinates.
    a : float
        The coefficient a in the model problem.
    kappa_0 : float
        The coefficient k_0 in the model problem.
    kappa_L : float
        The coefficient k_L in the model problem.

    Returns
    -------
    np.ndarray
        The global stiffness matrix
    """
    n = len(x) - 1 # number of subintervals
    A = np.zeros((n+1, n+1)) # initialize the global stiffness matrix

    for i in range(n):
        h = x[i+1] - x[i]
        xmid = (x[i+1] + x[i])/2 # interval midpoint
        amid = a(xmid) # value of a(x) at the midpoint

        A[i,i] += amid/h
        A[i,i+1] -= amid/h
        A[i+1,i] -= amid/h
        A[i+1,i+1] += amid/h

    # Add R matrix
    A[0,0] += kappa_0
    A[n,n] += kappa_L

    return A

def SourceAssembler1D(x : np.ndarray, f : callable, kappa_0 : float, kappa_L : float, g_0 : float, g_L : float):
    """
    Routine to assemble the source vector for the 1D case.
    
    Parameters
    ----------
    x : np.ndarray
        The vector (x_0, x_1, ..., x_N) of the nodes coordinates.
    f : callable
        The function f(x) in the model problem.
    kappa_0 : float
        The coefficient k_0 in the model problem.
    kappa_L : float
        The coefficient k_L in the model problem.
    g_0 : float
        The value g_0 in the model problem.
    g_L : float
        The value g_L in the model problem.

    Returns
    -------
    np.ndarray
        The global source vector
    """
    b = LoadAssembler1D(x, f)
    b[0] += kappa_0 * g_0
    b[-1] += kappa_L * g_L

    return b


def AdaptiveMethod1D(x : np.ndarray, f : callable, tol : float) -> tuple[np.ndarray, np.ndarray]:
    """
    Routine to evalute a FEM solution using the adaptive method.

    Parameters
    ----------
    x : np.ndarray
        The vector (x_0, x_1, ..., x_N) of the nodes coordinates.
    f : callable
        The function f(x) in the model problem.
    tol : float
        The tolerance for the adaptive method.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The solution vector and the nodes coordinates.
        The first element is the element residual eta(u_h)
        The second element is a new mesh that satisfies the tolerance.
    """
    n = len(x) - 1
    # Initialze element residuals (length is number of subintervals)
    eta = np.zeros(n)

    # Initialize refined mesh
    refined_mesh = x.copy()


    # Generate all element residuals
    for i in range(n):
        h = x[i+1] - x[i] # element length
        a = f(x[i]) # temporary values
        b = f(x[i+1])

        t = (a**2 + b**2) * h / 2 # integrate f^2 over the element. Trapezoidal rule
        eta[i] = h*np.sqrt(t)

        if eta[i] > tol:
            refined_mesh = np.append(refined_mesh, (x[i+1] + x[i])/2)

    return eta, np.sort(refined_mesh, axis=None)





