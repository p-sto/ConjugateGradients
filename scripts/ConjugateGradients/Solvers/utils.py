"""Miscellaneous functions"""

from typing import Optional
from Solvers.CG.cg_solver import ConjugateGradientMethodSolver
from solver import IterativeSolver

import numpy as np


def get_solver(name: str) -> Optional[IterativeSolver]:
    """Return solver based on name."""
    if name == 'CG':
        return ConjugateGradientMethodSolver
    raise NotImplementedError('Provided name for solver is wrong.')


def get_test_matrix(size: int = 50) -> np.matrix:
    """Return size by size test diagonal matrix."""
    test_matrix = np.zeros((size, size))
    return np.diagonal(test_matrix)
