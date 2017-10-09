"""
Contains implementation of Conjugate Gradient Method solver.
For more information search:
https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
"""

import copy
from typing import Tuple
from Solvers.PCG.preconditioners import jacobi
from Solvers.mixins import Convergence
from Solvers.solver import IterativeSolver

import numpy as np


class PreConditionedConjugateGradientSolver(IterativeSolver, Convergence):
    """Implements Preconditioned Conjugate Gradient method to solve system of linear equations."""

    def __init__(self, *args, preconditioner=jacobi, **kwargs) -> None:
        """Initialize PCG solver object, sets default pre-conditioner."""
        super(PreConditionedConjugateGradientSolver, self).__init__(*args, **kwargs)
        self.preconditioner = preconditioner

    def solve(self) -> Tuple[np.matrix, int]:
        """Solve system of linear equations."""
        i = 0
        x_vec = copy.deepcopy(self.x_vec)
        residual = self.b_vec - self.a_matrix * x_vec
        div = self.preconditioner(self.a_matrix, residual)
        delta_new = residual.T * div

        while i < self.max_iter and np.linalg.norm(residual) > self.tolerance:
            q_vec = self.a_matrix * div
            alpha = float(delta_new/(div.T*q_vec))
            # numpy has some problems with casting when using += notation...
            x_vec = x_vec + alpha*div
            residual = residual - alpha*q_vec
            s_pre = self.preconditioner(self.a_matrix, residual)
            delta_old = delta_new
            delta_new = residual.T*s_pre
            beta = delta_new/delta_old
            div = s_pre + float(beta)*div
            i += 1
        return x_vec, i
