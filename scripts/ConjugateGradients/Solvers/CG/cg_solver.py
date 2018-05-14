"""
Contains implementation of Conjugate Gradient Method solver.
For more information search:
https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
"""

import copy
from typing import Tuple
from scripts.ConjugateGradients.Solvers.common import IterativeSolver

import numpy as np


class ConjugateGradientSolver(IterativeSolver):
    """Implements Conjugate Gradient method to solve system of linear equations."""

    def __init__(self, *args, **kwargs):
        """Initialize CG solver object."""
        super(ConjugateGradientSolver, self).__init__(*args, **kwargs)
        self.name = 'CG'

    def solve(self) -> Tuple[np.matrix, np.matrix]:
        """Solve system of linear equations."""
        i = 0
        x_vec = copy.deepcopy(self.x_vec)
        residual = self.b_vec - self.a_matrix * x_vec
        div = residual
        delta_new = residual.T * residual

        while i < self.max_iter and np.linalg.norm(residual) > self.tolerance:
            q_vec = self.a_matrix * div
            alpha = float(delta_new/(div.T*q_vec))
            # numpy has some problems with casting when using += notation...
            x_vec = x_vec + alpha*div
            residual = residual - alpha*q_vec
            delta_old = delta_new
            delta_new = residual.T*residual
            beta = delta_new/delta_old
            div = residual + float(beta)*div
            self._register_residual(residual)
            i += 1
        self._finished_iter = i     # pylint: disable=attribute-defined-outside-init
        return x_vec, residual
