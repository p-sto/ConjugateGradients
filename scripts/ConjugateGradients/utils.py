"""Miscellaneous functions"""

from scripts.ConjugateGradients.Solvers.CG.cg_solver import ConjugateGradientSolver
from scripts.ConjugateGradients.Solvers.PCG.pcg_solver import PreConditionedConjugateGradientSolver
from scripts.ConjugateGradients.Solvers.common import IterativeSolver
from random import uniform

import numpy as np


def get_solver(name: str = None) -> IterativeSolver:
    """Return solver based on name."""
    if not name or name == 'CG':
        return ConjugateGradientSolver
    if name == 'PCG':
        return PreConditionedConjugateGradientSolver
    raise NotImplementedError('Provided name for solver is wrong.')


def get_test_matrix_three_diagonal(size: int = 50) -> np.matrix:
    """Return size by size test 3 diagonal matrix."""
    upper_diagonal = np.diagflat([1] * (size - 1), 1)
    lower_diagonal = np.diagflat([1] * (size - 1), -1)
    diagonal = np.diagflat([10 for x in range(size)])
    return np.matrix(lower_diagonal + diagonal + upper_diagonal)


def get_test_random_matrix_multi_diagonal(size: int = 50) -> np.matrix:
    """Return size by size random test 5 diagonal matrix."""
    upper_diagonal = np.diagflat([1] * (size - 1), 1)
    lower_diagonal = np.diagflat([1] * (size - 1), -1)
    diagonal = np.diagflat([10 for x in range(size)])
    up_rand_diag = np.diagflat([uniform(0, 1)] * (size - 30), 30)
    low_rand_diag = np.diagflat([uniform(0, 1)] * (size - 30), -30)
    return np.matrix(low_rand_diag + lower_diagonal + diagonal + upper_diagonal + up_rand_diag)


def get_test_matrix_diagonal(size: int = 50) -> np.matrix:
    """Return size by size test diagonal matrix."""
    diagonal = np.diagflat([10 for x in range(size)])
    return np.matrix(diagonal)
