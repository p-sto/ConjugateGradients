"""Reference script for implementation of Conjugate Gradient method."""

from random import uniform
from typing import cast

from scripts.ConjugateGradients.test_matrices import TestMatrices
from scripts.ConjugateGradients.utils import get_solver

import numpy as np


def main():
    """This is ``main`` script function - implemented only to test algorithm."""

    matrix_size = 100
    # patterns are: quadratic, rectangular, arrow, noise, curve
    # pattern='qra' means that testing matrix will be composition of quadratic, rectangular and arrow patterns
    a_matrix = TestMatrices.get_random_test_matrix(matrix_size, pattern='q')
    x_vec = np.vstack([1 for _ in range(matrix_size)])
    b_vec = np.vstack([uniform(0, 1) for _ in range(matrix_size)])
    cg_solver_cls = cast(callable, get_solver('CG'))
    pcg_solver_cls = cast(callable, get_solver('PCG'))
    cg_solver = cg_solver_cls(a_matrix, b_vec, x_vec)
    results_cg = cg_solver.solve()
    residual1 = results_cg[1]
    cg_solver.show_convergence_profile()

    pcg_solver = pcg_solver_cls(a_matrix, b_vec, x_vec)
    results_pcg = pcg_solver.solve()
    residual2 = results_pcg[1]

    print('CG solved in {0} iterations with residual lin norm = {1:.8f}'.format(cg_solver.finished_iter, np.linalg.norm(residual1)))
    print('PCG (Jacobi) solved in {0} iterations with residual lin norm = {1:.8f}'.format(pcg_solver.finished_iter, np.linalg.norm(residual2)))

    cg_solver_cls.compare_convergence_profiles(cg_solver, pcg_solver)


if __name__ == '__main__':
    main()
