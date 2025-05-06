import sys
import numpy as np
from matplotlib.tri import Triangulation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton
)
from PyQt5.QtCore import QTimer
import matplotlib.pyplot as plt
from tqdm import trange

from chapter4.main import HatGradients
from chapter3.main import PolyArea

# ---- FEM helper functions ----

def initmesh(g, hmax=0.05):
    xmin, xmax, ymin, ymax = g
    nx = max(3, int(np.ceil((xmax - xmin)/hmax)))
    ny = max(3, int(np.ceil((ymax - ymin)/hmax)))
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x, y)
    p = np.vstack((xx.flatten(), yy.flatten()))
    tri = Triangulation(p[0], p[1])
    t = tri.triangles.T
    edges = set()
    for tri_nodes in t.T:
        for i in range(3):
            e = tuple(sorted((tri_nodes[i], tri_nodes[(i+1)%3])))
            edges.add(e)
    e = np.array([list(e) for e in edges]).T
    return p, t, e

def assemble_matrices(p, t, v_field, diffusion_coef):
    n = p.shape[1]
    NT = t.shape[1]
    M = np.zeros((n,n))
    A = np.zeros((n,n))

    # Mass matrix
    for k in range(NT):
        nodes = t[:,k]
        x,y = p[0,nodes], p[1,nodes]
        area = PolyArea(x,y)
        localM = np.array([[2,1,1],[1,2,1],[1,1,2]]) * (area/12)
        M[np.ix_(nodes,nodes)] += localM

    # Stiffness + Advection
    for k in range(NT):
        nodes = t[:,k]
        x,y = p[0,nodes], p[1,nodes]
        xc,yc = x.mean(), y.mean()
        vx,vy = v_field(xc,yc)
        diff = diffusion_coef(xc,yc) if callable(diffusion_coef) else diffusion_coef
        area, b, c = HatGradients(x,y)

        Kdiff = diff*(np.outer(b,b)+np.outer(c,c))*area
        Kadv  = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                Kadv[i,j] = (vx*b[j] + vy*c[j])*(area/3)

        A[np.ix_(nodes,nodes)] += Kdiff + Kadv

    return M, A

def apply_dirichlet(p : np.ndarray, mat : np.ndarray, vec : np.ndarray) -> None:
    """
    Modify the matrix and vector to apply Dirichlet boundary conditions inplace

    Parameters
    ----------
    p : np.ndarray
        The array of nodes coordinates, shape (2, num_nodes).
    mat : np.ndarray
        The system matrix, shape (num_nodes, num_nodes).
    vec : np.ndarray
        The right-hand side vector, shape (num_nodes,).
    
    """
    xmin,xmax = p[0].min(), p[0].max()
    ymin,ymax = p[1].min(), p[1].max()
    tol = 1e-12
    bnodes = np.where(
        np.isclose(p[0],xmin,tol) |
        np.isclose(p[0],xmax,tol) |
        np.isclose(p[1],ymin,tol) |
        np.isclose(p[1],ymax,tol)
    )[0]
    for i in bnodes:
        mat[i,:] = 0
        mat[i,i] = 1
        vec[i]    = 0

# ---- PyQt5 GUI ----

class FEMWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FEM Advection–Diffusion (PyQt5)")

        # Central widget + layout
        cw = QWidget()
        self.setCentralWidget(cw)
        layout = QVBoxLayout(cw)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Controls
        btns = QHBoxLayout()
        layout.addLayout(btns)
        self.start_btn = QPushButton("Start")
        self.pause_btn = QPushButton("Pause")
        btns.addWidget(self.start_btn)
        btns.addWidget(self.pause_btn)
        self.start_btn.clicked.connect(self.start)
        self.pause_btn.clicked.connect(self.pause)
        self.pause_btn.setEnabled(False)

        # Simulation parameters
        self.dt      = 0.01
        self.T_final = 20.0
        self.steps   = int(self.T_final/self.dt)
        self.hmax    = 0.03
        self.diffcoef= 0.005
        self.vfield  = lambda x,y: (-y+0.5, x-0.5)

        # Prepare FEM
        self._prepare_sim()

        # Timer for stepping
        self.timer = QTimer()
        self.timer.setInterval(int(self.dt*1000))  # ms
        self.timer.timeout.connect(self._step)

    def _prepare_sim(self):
        # Mesh + IC
        self.p, self.t, _ = initmesh([0,1,0,1], hmax=self.hmax)
        self.u = np.exp(-50*((self.p[0]-0.2)**2 + (self.p[1]-0.2)**2))

        # Assemble M, A, precompute LHS
        M, A = assemble_matrices(self.p, self.t, self.vfield, self.diffcoef)
        self.M   = M
        self.A   = A
        self.LHS = M/self.dt + A

        # Initial plot
        self.tri  = Triangulation(self.p[0], self.p[1], self.t.T)
        self.plot = self.ax.tripcolor(self.tri, self.u,
                                      shading='gouraud', cmap='viridis')
        self.ax.set_title("t = 0.00")
        self.ax.set_aspect('equal')
        self.fig.colorbar(self.plot, ax=self.ax, label='u')

        self.current_step = 0

    def start(self):
        if not self.timer.isActive():
            self.timer.start()
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)

    def pause(self):
        if self.timer.isActive():
            self.timer.stop()
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)

    def _step(self):
        if self.current_step >= self.steps:
            self.pause()
            return

        rhs = self.M.dot(self.u)/self.dt
        apply_dirichlet(self.p, self.LHS, rhs)
        self.u = np.linalg.solve(self.LHS, rhs)

        self.current_step += 1
        t_now = self.current_step * self.dt
        self.plot.set_array(self.u)
        self.ax.set_title(f"t = {t_now:.2f}")
        self.canvas.draw_idle()

def main():
    app = QApplication(sys.argv)
    win = FEMWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
