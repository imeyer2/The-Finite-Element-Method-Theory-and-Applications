"""
    Chapter 4: The Finite Element Method in 2D

    Assume the model problem is

    - grad . (a grad u) = f  in Omega
    -n . (a grad u) = k(u-g_D) - g_N  on boundary of Omega

    where a, f, kappa > 0, g_D, g_N are a given functions (FUNCTIONS)
"""

import numpy as np
from chapter3.main import PolyArea

def HatGradients(x : np.ndarray, y : np.ndarray):
    """
    Routine to calculate gradients of the hat functions in 2D given the coordinates of the vertices.

    From the textbook:
    
    grad phi_i = (b_i, c_i)

    where the hat function is  phi_i(x,y) = a_i + b_i*x + c_i*y

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the vertices.
    y : np.ndarray
        The y-coordinates of the vertices.
    
    """

    area = PolyArea(x, y)
    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / (2*area)
    c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / (2*area)
    return area, b, c

def StiffnessAssembler2D(p : np.ndarray, t : np.ndarray, a : callable) -> np.ndarray:
    """
    Generate the stiffneff matrix for the 2D case model problem 

    - grad . (a grad u) = f  in Omega
    -n . (a grad u) = k(u-g_D) - g_N  on boundary of Omega

    where a, f, kappa > 0, g_D, g_N are a given functions (FUNCTIONS)


    Parameters
    ----------
    p : np.ndarray
        The array of nodes coordinates. Needs to be shape (2, num_nodes).
    t : np.ndarray
        The array of triangles. Needs to be shape (3, num_triangles)
    a : callable
        The function a(x,y) in the model problem.
    
    Returns
    -------
    np.ndarray
        The global stiffness matrix.

    """

    assert p.shape[0] == 2, "The array of nodes coordinates must have shape (2, num_nodes)"
    assert t.shape[0] == 3, "The array of triangles must have shape (3, num_triangles)"

    num_points = p.shape[1] # number of nodes
    num_elements = t.shape[1] # number of triangles (elements)

    A = np.zeros((num_points, num_points)) # initialize the global matrix

    for K in range(num_elements): # loop over the triangles
        loc2glb = t[:, K] # local-to-global map. The K-th column of t (i.e., the K-th triangle)
        x = p[0, loc2glb] # global x coordinates of the nodes
        y = p[1, loc2glb] # global y coordinates of the nodes

        area, b, c = HatGradients(x, y) # triangle area


        xc = np.mean(x) # x-coordinate of the centroid
        yc = np.mean(y) # y-coordinate of the centroid

        abar = a(xc, yc) # value of a(x,y) at the centroid

        AK = abar * area * (np.outer(b, b) + np.outer(c, c)) # local stiffness matrix 
     
        A[loc2glb, loc2glb] += AK # add the local stiffness matrix to the global stiffness matrix

    return A

def RobinMassMatrix2D(p : np.ndarray, e : np.ndarray, kappa : callable) -> np.ndarray:
    """
    Generate the mass matrix for the Robin boundary condition in 2D.
    
    Parameters
    ----------
    p : np.ndarray
        The array of nodes coordinates. Needs to be shape (2, num_nodes).
    e : np.ndarray
        The array of edges. Needs to be shape (2, num_edges)
    kappa : callable
        The function kappa(x,y) in the model problem.
    
    Returns
    -------
    np.ndarray
        The global robin mass matrix.

    See Also
    --------
    StiffnessAssembler2D
    """

    assert p.shape[0] == 2, "The array of nodes coordinates must have shape (2, num_nodes)"
    assert e.shape[0] == 2, "The array of edges must have shape (2, num_edges)"

    num_points = p.shape[1] # number of nodes
    num_edges = e.shape[1] # number of edges

    R = np.zeros((num_points, num_points)) # initialize the global matrix

    for E in range(num_edges):
        loc2glb = e[:, E]
        x = p[0, loc2glb] # node x-coordinates
        y = p[1, loc2glb] # node y-coordinates

        length = np.linalg.norm([x[0] - x[1], y[0] - y[1]]) # edge length
        xc = np.mean(x) # x-coordinate of the centroid
        yc = np.mean(y) # y-coordinate of the centroid
        k = kappa(xc, yc) # value of kappa(x,y) at the centroid

        RE = np.array([[2, 1], [1, 2]]) * k * length / 6
        R[loc2glb, loc2glb] += RE 

    return R

def RobinLoadVector2D(p : np.ndarray, e : np.ndarray, kappa : callable, g_D : callable, g_N : callable) -> np.ndarray:
    """
    Generate the load vector for the Robin boundary condition in 2D.
    
    Parameters
    ----------
    p : np.ndarray
        The array of nodes coordinates. Needs to be shape (2, num_nodes).
    e : np.ndarray
        The array of edges. Needs to be shape (2, num_edges)
    kappa : callable
        The function kappa(x,y) in the model problem.
    g_D : callable
        The function g_D(x,y) in the model problem.
    g_N : callable
        The function g_N(x,y) in the model problem.
    
    Returns
    -------
    np.ndarray
        The boundary vector (r) load vector.

    See Also
    --------
    StiffnessAssembler2D
    """

    assert p.shape[0] == 2, "The array of nodes coordinates must have shape (2, num_nodes)"
    assert e.shape[0] == 2, "The array of edges must have shape (2, num_edges)"

    num_points = p.shape[1] # number of nodes
    num_edges = e.shape[1] # number of edges
    

    r = np.zeros(num_points) # initialize the global vector

    for E in range(num_edges):
        loc2glb = e[:, E]
        x = p[0, loc2glb] # node x-coordinates
        y = p[1, loc2glb] # node y-coordinates

        length = np.linalg.norm([x[0] - x[1], y[0] - y[1]]) # edge length
        xc = np.mean(x) # x-coordinate of the centroid
        yc = np.mean(y) # y-coordinate of the centroid
        k = kappa(xc, yc) # value of kappa(x,y) at the centroid
        gd = g_D(xc, yc) # value of g_D(x,y) at the centroid
        gn = g_N(xc, yc) # value of g_N(x,y) at the centroid

        tmp = k*gd+gn

        rE : np.ndarray = tmp * np.array([1, 1]) * length / 2
        r[loc2glb] += rE

    return r

def RivaraRefinement2D(p: np.ndarray, t: np.ndarray) -> tuple:
    """
    Routine to refine the mesh using the Rivara algorithm.

    For triangles that are too erroneous, the algorithm splits them by 
    drawing a median from the longest edge to the opposite vertex.

    Parameters
    ----------
    p : np.ndarray
        The array of nodes coordinates. Needs to be shape (2, num_nodes).
    t : np.ndarray
        The array of triangles. Needs to be shape (3, num_triangles).

    Returns
    -------
    tuple
        The refined mesh (p, t).

    """
    num_triangles = t.shape[1]  # number of triangles (elements)
    new_points = p.copy()
    new_triangles = []

    for K in range(num_triangles):
        loc2glb = t[:, K]
        x = p[0, loc2glb] # x coordinates of triangles
        y = p[1, loc2glb] # y coordinates of triangles

        # Compute edge lengths
        edge_lengths = [
            np.linalg.norm([x[1] - x[0], y[1] - y[0]]),
            np.linalg.norm([x[2] - x[1], y[2] - y[1]]),
            np.linalg.norm([x[0] - x[2], y[0] - y[2]])
        ]
        longest_edge_idx = np.argmax(edge_lengths)

        # Find the midpoint of the longest edge
        if longest_edge_idx == 0:
            midpoint = (p[:, loc2glb[0]] + p[:, loc2glb[1]]) / 2
            opposite_vertex = loc2glb[2]
        elif longest_edge_idx == 1:
            midpoint = (p[:, loc2glb[1]] + p[:, loc2glb[2]]) / 2
            opposite_vertex = loc2glb[0]
        else:
            midpoint = (p[:, loc2glb[2]] + p[:, loc2glb[0]]) / 2
            opposite_vertex = loc2glb[1]

        # Add the midpoint to the list of points
        midpoint_idx = new_points.shape[1]
        new_points = np.hstack((new_points, midpoint.reshape(2, 1)))

        # Create new triangles
        for i in range(3):
            if i != longest_edge_idx:
                new_triangle = [loc2glb[i], loc2glb[(i + 1) % 3], midpoint_idx]
                new_triangles.append(new_triangle)
        new_triangles.append([midpoint_idx, loc2glb[longest_edge_idx], opposite_vertex])

    # Convert new_triangles to a numpy array
    new_triangles = np.array(new_triangles).T

    return new_points, new_triangles
