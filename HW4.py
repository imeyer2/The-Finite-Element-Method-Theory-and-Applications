from chapter4.main import StiffnessAssembler2D
from chapter3.main import MassAssembler2D

import numpy as np


stiff = StiffnessAssembler2D(p = np.array([[0, 0], [1, 0], [0, 1]]).T, 
                             t = np.array([[0, 1, 2]]).T, 
                             a = lambda x, y: 1)

mass = MassAssembler2D(p = np.array([[0, 0], [1, 0], [0, 1]]).T,
                        t = np.array([[0, 1, 2]]).T)

print("Stiffness matrix:\n", stiff)
print("Mass matrix:\n", mass) 