"""
    Chapter 3: Piecewise Polynomial Approximations in 2D

    

"""
import numpy as np


def PolyArea(x : np.ndarray, y : np.ndarray) -> float:
    """
    Numpy implementation of shoelace theorem to calculate the area of a polygon.

    Note: Avoiding a for loop is ~50x faster.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the polygon.
    y : np.ndarray
        The y-coordinates of the polygon.

    Returns
    -------
    float
        The area of the polygon.
    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def MassAssembler2D(p : np.ndarray, t : np.ndarray) -> np.ndarray:
    """
    Routine to assemble the mass matrix for the 2D case.

    Parameters
    ----------
    p : np.ndarray
        The array of nodes coordinates. Needs to be shape (2, num_nodes).
    t : np.ndarray
        The array of triangles. Needs to be shape (3, num_triangles)

    Returns
    -------
    np.ndarray
        The global matrix.
    """

    assert p.shape[0] == 2, "The array of nodes coordinates must have shape (2, num_nodes)"
    assert t.shape[0] == 3, "The array of triangles must have shape (3, num_triangles)"


    num_points = p.shape[1] # number of nodes
    num_elements = t.shape[1] # number of triangles (elements)

    M = np.zeros((num_points, num_points)) # initialize the global matrix


    for K in range(num_elements): # loop over the triangles
        loc2glb = t[:, K] # local-to-global map. The K-th column of t (i.e., the K-th triangle)
        x = p[0, loc2glb] # global x coordinates of the nodes
        y = p[1, loc2glb] # global y coordinates of the nodes

        area = PolyArea(x, y) # triangle area

        MK = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])/12 * area # local mass matrix
        
        M[loc2glb, loc2glb] += MK # add the local mass matrix to the global mass matrix

    return M

def LoadAssembler2D(p : np.ndarray, t : np.ndarray, f : callable) -> np.ndarray:


    assert p.shape[0] == 2, "The array of nodes coordinates must have shape (2, num_nodes)"
    assert t.shape[0] == 3, "The array of triangles must have shape (3, num_triangles)"


    num_points = p.shape[1] # number of nodes
    num_elements = t.shape[1] # number of triangles (elements)

    b = np.zeros(num_points) # initialize the global vector

    for K in range(num_elements):
        loc2glb = t[:, K]
        x = p[0, loc2glb]
        y = p[1, loc2glb]
        area = PolyArea(x, y)
        bK = np.array([f(x[0], y[0]), f(x[1], y[1]), f(x[2], y[2])]) * area / 3

        b[loc2glb] += bK

    return b

def L2Projector2D(p : np.ndarray, t : np.ndarray, f : callable) -> np.ndarray:
    """
    Routine to compute the L2 projection of a function f in 2D.

    Parameters
    ----------
    p : np.ndarray
        The array of nodes coordinates. Needs to be shape (2, num_nodes)."
    t : np.ndarray
        The array of triangles. Needs to be shape (3, num_triangles)
    f : callable
        The function f(x, y) to be approximated."
    """
    
    M = MassAssembler2D(p, t)
    b = LoadAssembler2D(p, t, f)

    Pf = np.linalg.solve(M, b)

    return Pf