"""Reference script for implementation of Conjugate Gradient method."""

from random import uniform
from scripts.ConjugateGradients.utils import get_test_random_matrix_multi_diagonal, get_solver

import numpy as np


def main():
    """This is ``main`` script function - implemented only to test algorithm."""

    matrix_size = 100
    a_matrix = get_test_random_matrix_multi_diagonal(matrix_size)
    x_vec = np.vstack([1 for x in range(matrix_size)])
    b_vec = np.vstack([uniform(0, 1) for x in range(matrix_size)])
    CGSolver = get_solver('CG')             # pylint: disable=invalid-name; get_solver returns Class
    PCGJacobiSolver = get_solver('PCG')     # pylint: disable=invalid-name; get_solver returns Class
    cg_solver = CGSolver(a_matrix, b_vec, x_vec)
    results_cg = cg_solver.solve()
    residual1 = results_cg[1]
    cg_solver.show_convergence_profile()

    pcg_solver = PCGJacobiSolver(a_matrix, b_vec, x_vec)
    results_pcg = pcg_solver.solve()
    residual2 = results_pcg[1]

    print('CG solved in {0} iterations with residual lin norm = {1:.8f}'.format(cg_solver.finished_iter, np.linalg.norm(residual1)))
    print('PCG (Jacobi) solved in {0} iterations with residual lin norm = {1:.8f}'.format(pcg_solver.finished_iter, np.linalg.norm(residual2)))

    CGSolver.compare_convergence_profiles(cg_solver, pcg_solver)


if __name__ == '__main__':
    main()
