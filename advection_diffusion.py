import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from chapter4.main import HatGradients
from chapter3.main import PolyArea


def initmesh(g, hmax=0.05):
    """
    Initialize a rectangular mesh.
    
    Parameters
    ----------
    g : list
        The domain boundaries [xmin, xmax, ymin, ymax].
    hmax : float
        The maximum edge length.
        
    Returns
    -------
    tuple
        (p, t, e) - points, triangles, edges
    """
    if isinstance(g, list) and len(g) == 4:
        xmin, xmax, ymin, ymax = g
        nx = max(3, int(np.ceil((xmax - xmin) / hmax)))
        ny = max(3, int(np.ceil((ymax - ymin) / hmax)))
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        xx, yy = np.meshgrid(x, y)
        p = np.vstack((xx.flatten(), yy.flatten()))
        tri = Triangulation(p[0], p[1])
        t = tri.triangles.T

        edges = set()
        for tri_nodes in t.T:
            for i in range(3):
                edge = tuple(sorted([tri_nodes[i], tri_nodes[(i+1)%3]]))
                edges.add(edge)

        e = np.array([list(edge) for edge in edges]).T
        return p, t, e

    else:
        raise NotImplementedError("Only rectangular domains are supported")


def AdvectionDiffusionAssembler2D(p, t, u, v_field, diffusion_coef):
    """
    Assemble the system matrix for advection-diffusion equation.
    
    Parameters
    ----------
    p : np.ndarray
        The array of nodes coordinates, shape (2, num_nodes).
    t : np.ndarray
        The array of triangles, shape (3, num_triangles).
    u : np.ndarray
        The current solution vector.
    v_field : callable
        Function that returns the velocity field (vx, vy) at a point (x, y).
    diffusion_coef : float or callable
        Diffusion coefficient, can be constant or function of position.
        
    Returns
    -------
    tuple
        (A, b) - system matrix and right-hand side vector
    """
    num_nodes = p.shape[1]
    nt = t.shape[1]
    A = np.zeros((num_nodes, num_nodes))  # System matrix
    b = np.zeros(num_nodes)               # Right-hand side vector

    for tri_idx in tqdm(range(nt), desc="Assembling system", total=nt):
        nodes = t[:, tri_idx]
        x = p[0, nodes]
        y = p[1, nodes]

        xc, yc = x.mean(), y.mean()
        
        # Get diffusion coefficient
        if callable(diffusion_coef):
            diff_val = diffusion_coef(xc, yc)
        else:
            diff_val = diffusion_coef
            
        # Get velocity field
        vx, vy = v_field(xc, yc)
        
        area, b_coefs, c_coefs = HatGradients(x, y)
        
        # Diffusion part (similar to stiffness matrix)
        diffusion_matrix = diff_val * (np.outer(b_coefs, b_coefs) + np.outer(c_coefs, c_coefs)) * area
        
        # Advection part 
        # Construct local advection matrix using test and basis functions
        advection_matrix = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                # vx * phi_i * d(phi_j)/dx + vy * phi_i * d(phi_j)/dy
                # Here we approximate phi_i as 1/3 at the centroid
                advection_matrix[i, j] = (vx * b_coefs[j] + vy * c_coefs[j]) * (1/3) * area
        
        # Add local matrices to global matrix
        A[np.ix_(nodes, nodes)] += diffusion_matrix + advection_matrix
        
        # For this example, we don't have a source term, but could add one here
        # b[nodes] += source_term(xc, yc) * area / 3

    return A, b


def set_boundary_conditions(p, A, b, boundary_type, boundary_value=None):
    """
    Apply boundary conditions.
    
    Parameters
    ----------
    p : np.ndarray
        The array of nodes coordinates, shape (2, num_nodes).
    A : np.ndarray
        The system matrix.
    b : np.ndarray
        The right-hand side vector.
    boundary_type : str
        Type of boundary condition ('dirichlet' or 'neumann').
    boundary_value : float or callable, optional
        The value for Dirichlet boundary or function for Neumann boundary.
        
    Returns
    -------
    tuple
        (A_mod, b_mod) - modified matrix and vector with boundary conditions applied
    """
    xmin, xmax = p[0].min(), p[0].max()
    ymin, ymax = p[1].min(), p[1].max()
    tol = 1e-12
    
    # Find boundary nodes
    boundary_nodes = np.where(
        (np.isclose(p[0], xmin, atol=tol)) |
        (np.isclose(p[0], xmax, atol=tol)) |
        (np.isclose(p[1], ymin, atol=tol)) |
        (np.isclose(p[1], ymax, atol=tol))
    )[0]
    
    A_mod = A.copy()
    b_mod = b.copy()
    
    if boundary_type.lower() == 'dirichlet':
        # Apply Dirichlet boundary conditions: u = boundary_value on boundary
        for node in boundary_nodes:
            A_mod[node, :] = 0
            A_mod[node, node] = 1
            
            if callable(boundary_value):
                b_mod[node] = boundary_value(p[0, node], p[1, node])
            else:
                b_mod[node] = boundary_value if boundary_value is not None else 0
    
    elif boundary_type.lower() == 'neumann':
        # For Neumann boundary conditions, we need to modify the right-hand side
        # This is a simplified version; full implementation would require boundary integrals
        pass
    
    return A_mod, b_mod


def solve_advection_diffusion(hmax=0.02,
                              diffusion_coef=0.01,
                              steady_state=True,
                              dt=0.01, T_final=1.0,
                              n_snapshots=5):
    # Generate mesh & velocity, initial condition…
    domain = [0, 1, 0, 1]
    p, t, e = initmesh(domain, hmax=hmax)
    num_nodes = p.shape[1]

    def velocity_field(x, y):
        return [-y + 0.5, x - 0.5]

    def initial_condition(x, y):
        return np.exp(-50 * ((x - 0.2)**2 + (y - 0.2)**2))

    u = np.array([initial_condition(px, py) 
                  for px, py in zip(p[0], p[1])])

    if not steady_state:
        num_steps = int(T_final / dt)
        # pick evenly‐spaced steps (including step 0)
        snapshot_steps = np.linspace(0, num_steps, n_snapshots, dtype=int)
        snapshots = []

        for step in tqdm(range(num_steps+1), desc="Time stepping"):
            if step in snapshot_steps:
                snapshots.append((step*dt, u.copy()))

            if step == num_steps:
                break

            # assemble mass matrix M
            M = np.zeros((num_nodes, num_nodes))
            for tri_idx in range(t.shape[1]):
                nodes = t[:, tri_idx]
                x = p[0, nodes]
                y = p[1, nodes]
                area = PolyArea(x, y)
                local_M = np.array([[2,1,1],
                                    [1,2,1],
                                    [1,1,2]]) * area/12
                M[np.ix_(nodes, nodes)] += local_M

            # assemble A, b
            A, b = AdvectionDiffusionAssembler2D(
                p, t, u, velocity_field, diffusion_coef
            )

            # implicit Euler: (M/dt + A) u_new = M/dt * u_old + b
            lhs = M/dt + A
            rhs = M.dot(u)/dt + b
            lhs_bc, rhs_bc = set_boundary_conditions(
                p, lhs, rhs, 'dirichlet', boundary_value=0
            )
            u = np.linalg.solve(lhs_bc, rhs_bc)

        # now plot snapshots side by side
        fig, axes = plt.subplots(1, n_snapshots,
                                 figsize=(4*n_snapshots, 4),
                                 squeeze=False)
        triang = Triangulation(p[0], p[1], t.T)
        for ax, (time, u_snap) in zip(axes[0], snapshots):
            cont = ax.tripcolor(triang, u_snap,
                                shading='gouraud', cmap='viridis')
            ax.set_title(f"t = {time:.2f}")
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

        # one colorbar for all
        fig.colorbar(cont, ax=axes[0], fraction=0.02,
                     pad=0.04, label='Concentration')
        plt.tight_layout()
        plt.show()

        return p, t, u


if __name__ == "__main__":
    # Example 1: Steady-state with low diffusion
    solve_advection_diffusion(hmax=0.02, diffusion_coef=0.01, steady_state=True)
    
    # Example 2: Time-dependent simulation
    solve_advection_diffusion(hmax=0.03, diffusion_coef=0.005, steady_state=False, dt=0.01, T_final=2.0)