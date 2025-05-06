import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from chapter4.main import HatGradients
from chapter3.main import PolyArea


def initmesh(g, hmax=0.05):
    xmin, xmax, ymin, ymax = g
    nx = max(3, int(np.ceil((xmax-xmin)/hmax)))
    ny = max(3, int(np.ceil((ymax-ymin)/hmax)))
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x, y)
    p = np.vstack((xx.flatten(), yy.flatten()))
    tri = Triangulation(p[0], p[1])
    t   = tri.triangles.T

    edges = set()
    for tri_nodes in t.T:
        for i in range(3):
            e = tuple(sorted((tri_nodes[i], tri_nodes[(i+1)%3])))
            edges.add(e)
    e = np.array([list(e) for e in edges]).T

    return p, t, e


def DiffusionAssembler2D(p, t, diffusion_coef):
    """
    Assemble the diffusion (stiffness) matrix:
      D_{ij} = ∫ (diff_val ∇φ_j · ∇φ_i) dΩ
    """
    N  = p.shape[1]
    NT = t.shape[1]
    D  = np.zeros((N, N))

    for K in range(NT):
        nodes = t[:,K]
        x, y = p[0,nodes], p[1,nodes]
        area, b, c = HatGradients(x, y)

        # diffusion_coef may be a function or constant
        xc, yc = x.mean(), y.mean()
        diff = diffusion_coef(xc, yc) if callable(diffusion_coef) else diffusion_coef

        localD = diff * (np.outer(b,b) + np.outer(c,c)) * area
        D[np.ix_(nodes, nodes)] += localD

    return D


def ConvectiveAssembler2D(p, t, v_field):
    """
    Assemble the convective matrix:
      C_{ij} = ∫ (v·∇φ_j) φ_i dΩ
    """
    N  = p.shape[1]
    NT = t.shape[1]
    C  = np.zeros((N, N))

    for K in range(NT):
        nodes = t[:,K]
        x, y = p[0,nodes], p[1,nodes]
        xc, yc = x.mean(), y.mean()
        area, b, c = HatGradients(x, y)

        vx, vy = v_field(xc, yc)
        vgrad = vx*b + vy*c               # shape (3,)
        localC = np.outer(np.ones(3), vgrad) * (area/3)

        C[np.ix_(nodes, nodes)] += localC

    return C


def set_dirichlet(p, A, b, boundary_value : float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Strongly impose the Dirichlet boundary condition:

    Parameters
    ----------
    p : np.ndarray
        The array of nodes coordinates, shape (2, num_nodes).
    A : np.ndarray
        The ENTIRE matrix (this is NOT the stiffness matrix), shape (num_nodes, num_nodes).
    b : np.ndarray
        The ENTIRE right-hand side vector, shape (num_nodes,). This vector represents the boundary values.

    Returns
    -------
    tuple
        A_bc : np.ndarray
            The modified matrix with Dirichlet boundary conditions applied, shape (num_nodes, num_nodes).
        b_bc : np.ndarray
            The modified right-hand side vector with Dirichlet boundary conditions applied, shape (num_nodes,).
    
    """
    # zero‐Dirichlet on the rectangle boundary
    xmin, xmax = p[0].min(), p[0].max()
    ymin, ymax = p[1].min(), p[1].max()
    tol = 1e-12
    bnodes = np.where(
        np.isclose(p[0], xmin, tol) |
        np.isclose(p[0], xmax, tol) |
        np.isclose(p[1], ymin, tol) |
        np.isclose(p[1], ymax, tol)
    )[0]

    for i in bnodes:
        A[i,:] = 0
        A[i,i] = 1
        b[i]   = boundary_value
    return A, b


def solve_advection_diffusion(
    hmax=0.02,
    diffusion_coef=0.01,
    steady_state=True,
    dt=0.01, T_final=1.0,
    n_snapshots=5
):
    # 1) Mesh and velocity
    domain = [0,1,0,1]
    p, t, e = initmesh(domain, hmax=hmax)
    N = p.shape[1]

    def v_field(x,y):
        return -y+0.5, x-0.5

    # 2) Pre‐assemble matrices
    D = DiffusionAssembler2D(p, t, diffusion_coef)
    C = ConvectiveAssembler2D(p, t, v_field)

    # 3) Initial condition
    # u = np.exp(-50*((p[0]-0.2)**2 + (p[1]-0.2)**2))
    u = np.zeros(N)

    if steady_state:
        # Steady solve: (D + C) u = 0, with Dirichlet
        A = D + C
        b = np.zeros(N)
        A_bc, b_bc = set_dirichlet(p, A.copy(), b.copy())
        u = np.linalg.solve(A_bc, b_bc)

        # plot omitted…
        return p,t,u

    # 4) Time‐dependent: M, plus implicit Euler
    #    M_{ij}=∫φ_j φ_i dΩ
    M = np.zeros((N,N))
    for K in range(t.shape[1]):
        nodes = t[:,K]
        x, y = p[0,nodes], p[1,nodes]
        area = PolyArea(x,y)
        localM = np.array([[2,1,1],[1,2,1],[1,1,2]]) * (area/12)
        M[np.ix_(nodes, nodes)] += localM

    snapshots = []
    steps = int(T_final/dt)
    times = np.linspace(0, steps, n_snapshots, dtype=int)

    for k in tqdm(range(steps+1), desc="Time stepping"):
        if k in times:
            snapshots.append((k*dt, u.copy()))
        if k==steps:
            break

        # implicit Euler: (M/dt + D + C) u_{n+1} = M/dt * u_n
        LHS = M/dt + D + C
        RHS = M.dot(u)/dt
        LHS_bc, RHS_bc = set_dirichlet(p, LHS.copy(), RHS.copy(), boundary_value=10.0)
        u = np.linalg.solve(LHS_bc, RHS_bc)

    # 5) Plot side‐by‐side snapshots…
    fig, axes = plt.subplots(1, len(snapshots), figsize=(4*len(snapshots),4))
    triang = Triangulation(p[0], p[1], t.T)
    for ax, (t0, usnap) in zip(axes, snapshots):
        ax.tripcolor(triang, usnap, shading='gouraud', cmap='viridis')
        ax.set_title(f"t={t0:.2f}")
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    return p, t, u


if __name__=="__main__":
    solve_advection_diffusion(
      hmax=0.03, diffusion_coef=0.005,
      steady_state=False, dt=0.01, T_final=3.0
    )
