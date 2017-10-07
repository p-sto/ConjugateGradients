"""Stores implementation of IterativeSolver abstract class."""

from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np


class IterativeSolver(metaclass=ABCMeta):
    """Represents Iterative Solver interface."""
    tolerance = 1e-5

    def __init__(self, a_matrix: np.matrix, b_vec: np.vstack, x_vec: np.vstack, max_iter: int = 200) -> None:
        """Initialize solver

        :param a_matrix: Matrix for which we search solution.
        :param b_vec: Vector for which we search solution.
        :param x_vec: result vector
        :param max_iter:

        """
        self.a_matrix = a_matrix
        self.b_vec = b_vec
        self.x_vec = x_vec
        self.max_iter = max_iter
        self.residual_values = []   # type: list

    def _register_residual(self, res_val: float) -> None:
        """Register residual value for particular iteration."""
        self.residual_values.append(res_val)

    @abstractmethod
    def solve(self) -> Tuple[np.matrix, int]:
        """Solve system of linear equations."""
        raise NotImplementedError
