"""

    Chapter 5: Time Dependent Problems


    This chapter discusses PDE that depend on time and space. Observe that FEM, so far, is used to discretize the spatial domain.
    We can break down the FEM solution from the non-time 


    - The below is the FEM solution to NON time dependent PDE. Recall that our goal is to find the c_i. 
    This is why we use the Mass Matrix or Stiffness Matrix because we are solving for the weights in this linear combination.
    u = \sum_{i=1}^{N} c_i \phi_i(x) 


    - When time is introduced, these weights (i.e. the c_i) become functions of time. We call them \eta_i(t)

    u = \sum_{i=1}^{N} \eta_i(t) \phi_i(x)

"""

import numpy as np
from chapter2.main import StiffnessAssembler1D
from chapter1.main import MassAssembler1D, LoadAssembler1D
import matplotlib.pyplot as plt

# Section 5.5: Computer implementation

def BackwardEulerHeatSolver1D():
    h : float = 0.01 # mesh size
    x : np.ndarray = np.arange(0, 1, h) # mesh points (x axis)
    m : int = 100 # number of time steps
    T : float = 0.5 # final time
    t = np.linspace(0, T, m + 1) # time grid
    xi : np.ndarray = 0.5-np.abs(0.5-x) # initial condition (hat function)
    kappa = [1e6,1e6] # Robin BC to approximate Dirichlet BC of 0 
    g = [0,0]
    A = StiffnessAssembler1D(x, a = lambda z: 1, kappa_0 = kappa[0], kappa_L = kappa[1]) # stiffness matrix
    M = MassAssembler1D(x) # mass matrix
    b = LoadAssembler1D(x, f = lambda z: 2*z) # f=2x load vector

    # Create subplots
    num_plots = 5  # Number of subplots (e.g., every 25 steps)
    fig, axes = plt.subplots(1, num_plots, figsize=(15, 3), sharey=True)
    plot_indices = np.linspace(0, m, num_plots, dtype=int)  # Indices for subplots

    for l in range(m):  # loop over time steps
        k = t[l + 1] - t[l]  # time step
        xi = np.linalg.solve(M + k * A, M @ xi + k * b)

        # Plot at specific time steps
        if l in plot_indices:
            ax = axes[np.where(plot_indices == l)[0][0]]
            ax.plot(x, xi)
            ax.set_title(f't = {t[l + 1]:.2f}')
            ax.set_xlabel('x')
            ax.set_ylabel('u(x,t)')

    plt.tight_layout()
    plt.suptitle('Backward Euler Heat Solver 1D', y=1.05)
    plt.show()

if __name__ == "__main__":
    BackwardEulerHeatSolver1D()
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('Backward Euler Heat Solver 1D')
    plt.legend()
    plt.show()