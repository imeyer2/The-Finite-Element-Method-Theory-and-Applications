import numpy as np
import matplotlib.pyplot as plt
from chapter2.main import StiffnessAssembler1D
from chapter1.main import MassAssembler1D, LoadAssembler1D

# Exact solution series (truncate after N terms)
def exact_solution(x, t, N=25):
    series = np.zeros_like(x)
    for n in range(1, N+1):
        coeff = ((-1)**n - 1) / (n**2)
        series += coeff * np.sin(n * np.pi * x) * np.exp(n**2 * np.pi**2 * t / -10)
    return (4 / np.pi**3) * series

# Section 5.5: Backward Euler for u_t = u_xx, u(0)=u(1)=0
def BackwardEulerHeatSolver1D():
    h = 0.01  # mesh size
    x = np.arange(0, 1 + h, h)  # include endpoint
    num_steps = 100  # number of time steps
    T = 0.1  # final time
    t = np.linspace(0, T, num_steps + 1)  # time grid
    xi = x * (1 - x)  # initial condition

    # Assemble matrices
    A = StiffnessAssembler1D(x,
                              a=lambda z: 1/10,
                              kappa_0=1e6,
                              kappa_L=1e6)
    M = MassAssembler1D(x)
    b = LoadAssembler1D(x, f=lambda z: 0)  # zero load

    # Prepare subplots for snapshots
    num_plots = 5
    fig, axes = plt.subplots(1, num_plots, figsize=(15, 3), sharey=True)
    snapshot_indices = np.linspace(0, num_steps - 1, num_plots, dtype=int)

    # Time-stepping
    for l in range(num_steps):
        k = t[l + 1] - t[l]
        xi = np.linalg.solve(M + k * A, M @ xi + k * b)

        if l in snapshot_indices:
            idx = np.where(snapshot_indices == l)[0][0]
            ax = axes[idx]

            # numerical
            ax.plot(x, xi, label='Numerical')
            # exact
            u_ex = exact_solution(x, t[l+1])
            ax.plot(x, u_ex, '--', label = 'Exact')

            # sup error
            max_err = np.max(np.abs(xi - u_ex))
            ax.set_title(f't = {t[l+1]:.2f}, max err = {max_err:.2e}')

            ax.set_xlabel('x')
            if idx == 0:
                ax.set_ylabel('u(x,t)')

    plt.suptitle('Backward Euler Heat Solver 1D: Numerical vs Exact')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    BackwardEulerHeatSolver1D()
