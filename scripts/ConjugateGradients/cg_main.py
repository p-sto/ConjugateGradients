"""Reference script for implementation of Conjugate Gradient method."""

from random import uniform
from scripts.ConjugateGradients.utils import get_test_matrix_three_diagonal, get_solver

import numpy as np


def main():
    """This is ``main`` script function - implemented only to test algorithm."""

    matrix_size = 100
    a_matrix = get_test_matrix_three_diagonal(matrix_size)
    x_vec = np.vstack([1 for x in range(matrix_size)])
    b_vec = np.vstack([uniform(0, 1) for x in range(matrix_size)])
    Solver = get_solver('CG')      # pylint: disable=invalid-name; get_solver returns Class
    solver = Solver(a_matrix, b_vec, x_vec)
    x_vec, i = solver.solve()

    print('Solved in {} iterations'.format(i))

    solver.show_convergence_profile()


if __name__ == '__main__':
    main()
