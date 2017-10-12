"""Miscellaneous functions"""

from scripts.ConjugateGradients.Solvers.CG.cg_solver import ConjugateGradientSolver
from scripts.ConjugateGradients.Solvers.PCG.pcg_solver import PreConditionedConjugateGradientSolver
from scripts.ConjugateGradients.Solvers.common import IterativeSolver

import numpy as np


def get_solver(name: str=None) -> IterativeSolver:
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


def get_test_matrix_diagonal(size: int = 50) -> np.matrix:
    """Return size by size test diagonal matrix."""
    diagonal = np.diagflat([10 for x in range(size)])
    return np.matrix(diagonal)
