"""Miscellaneous functions"""

from typing import Optional
from Solvers.CG.cg_solver import ConjugateGradientSolver
from Solvers.PCG.pcg_solver import PreConditionedConjugateGradientSolver
from Solvers.solver import IterativeSolver

import numpy as np


def get_solver(name: str) -> Optional[IterativeSolver]:
    """Return solver based on name."""
    if name == 'CG':
        return ConjugateGradientSolver
    if name == 'PCG':
        return PreConditionedConjugateGradientSolver
    raise NotImplementedError('Provided name for solver is wrong.')


def get_test_matrix(size: int = 50) -> np.matrix:
    """Return size by size test 3 diagonal matrix."""
    upper_diagonal = np.diagflat([1] * (size - 1), 1)
    lower_diagonal = np.diagflat([1] * (size - 1), -1)
    diagonal = np.diagflat([10 for x in range(size)])
    return np.matrix(lower_diagonal + diagonal + upper_diagonal)
