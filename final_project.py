import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from chapter4.main import HatGradients

def initmesh(g, hmax=0.05):
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

        e = np.zeros((4, len(edges)), dtype=int)
        for i, (n1, n2) in enumerate(edges):
            e[0, i] = n1
            e[1, i] = n2
        return p, t, e

    else:
        raise NotImplementedError("Only rectangular domains are supported")

def JacResAssembler2D(p, e, t, u, Afcn, Ffcn):
    num_nodes = p.shape[1]
    nt = t.shape[1]
    J = np.zeros((num_nodes, num_nodes))
    r = np.zeros(num_nodes)

    for tri_idx in tqdm(range(nt), desc="Assembling Jacobian and Residual. Number of triangles", total=nt):
        nodes = t[:, tri_idx]
        x = p[0, nodes]
        y = p[1, nodes]

        xc, yc = x.mean(), y.mean()
        u_local = u[nodes]            # <-- now a 1D array of length 3
        uc = u_local.mean()

        area, b, c = HatGradients(x, y)
        ux = np.dot(u_local, b)
        uy = np.dot(u_local, c)

        tiny = 1e-8
        a_val = Afcn(uc)
        da_val = (Afcn(uc + tiny) - a_val) / tiny
        f_val  = Ffcn(xc, yc)

        # local residual (size 3) and local Jacobian (3×3)
        rK = ( a_val*(ux*b + uy*c)*area - f_val*(np.ones(3)*area/3) )
        JK = (
            a_val*(np.outer(b,b) + np.outer(c,c))
            + da_val*area/3*np.outer((ux*b + uy*c), np.ones(3))
        ) * area

        J[np.ix_(nodes, nodes)] += JK
        r[nodes] += rK

    # enforce u=0 on boundary
    xmin, xmax = p[0].min(), p[0].max()
    ymin, ymax = p[1].min(), p[1].max()
    tol = 1e-12
    bdy = np.where(
        (np.isclose(p[0], xmin, atol=tol)) |
        (np.isclose(p[0], xmax, atol=tol)) |
        (np.isclose(p[1], ymin, atol=tol)) |
        (np.isclose(p[1], ymax, atol=tol))
    )[0]

    for n in bdy:
        J[n, :] = 0
        J[n, n] = 1
        r[n]    = 0

    return J, r

def NewtonPoissonSolver2D(max_iters=200, tol=1e-6):
    g = [0,1,0,1]
    p, t, e = initmesh(g, hmax=0.01)
    num_nodes = p.shape[1]
    xi = np.zeros(num_nodes)   # 1-D initial guess

    for k in range(max_iters):
        J, r = JacResAssembler2D(
            p, e, t, xi,
            Afcn=lambda u: 0.125 + u**2,
            Ffcn=lambda x,y: 1.0   # constant source
        )
        # Newton step: J d = -r
        d = np.linalg.solve(J, -r)
        xi += d

        res_norm = np.linalg.norm(r)
        print(f"Iter {k+1}: ‖r‖ = {res_norm:.3e}, ‖d‖ = {np.linalg.norm(d):.3e}")
        if np.linalg.norm(d) < tol:
            print(f"--> converged in {k+1} iters.")
            break

    # Plotting
    fig = plt.figure(figsize=(14,6))

    ax1 = fig.add_subplot(1,2,1)
    triang = Triangulation(p[0], p[1], t.T)
    trip = ax1.tripcolor(triang, xi, edgecolors='k', cmap='viridis')
    fig.colorbar(trip, ax=ax1)
    ax1.set(aspect='equal', title='2D solution')

    ax2 = fig.add_subplot(1,2,2, projection='3d')
    ax2.plot_trisurf(p[0], p[1], xi, triangles=t.T, cmap='viridis', edgecolor='none')
    ax2.set(title='3D surface', xlabel='x', ylabel='y', zlabel='u')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    NewtonPoissonSolver2D()
