"""Reference script for implementation of Conjugate Gradient method."""

from Solvers.utils import get_test_matrix, get_solver

import numpy as np
from random import uniform

if __name__ == '__main__':
    size = 100
    A_matrix = get_test_matrix(size)
    x_vec = np.vstack([1 for x in range(size)])
    b_vec = np.vstack([uniform(0, 1) for x in range(size)])
    Solver = get_solver('CG')
    solver = Solver(A_matrix, b_vec, x_vec)
    x_vec, i = solver.solve()

    print('Solved in {} iterations'.format(i))
