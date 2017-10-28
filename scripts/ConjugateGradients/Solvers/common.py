"""Stores implementation of IterativeSolver abstract class."""

from abc import ABCMeta, abstractmethod
from typing import Tuple, List

import matplotlib.pyplot as plt
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
        self.name = ''
        if not self._is_pos_def:
            raise TypeError('Provided matrix is not positively defined.')
        self.a_matrix = a_matrix
        self.b_vec = b_vec
        self.x_vec = x_vec
        self.max_iter = max_iter
        self._finished_iter = 0
        self.residual_values = []   # type: list

    @property
    def finished_iter(self) -> int:
        """Return number of solver's iterations"""
        return self._finished_iter

    def _register_residual(self, conv: np.matrix) -> None:
        """Register residual value for particular iteration."""
        self.residual_values.append(np.linalg.norm(conv))

    def _is_pos_def(self) -> bool:
        """Check if matrix is positively defined using eigenvalues."""
        return np.all(np.linalg.eigvals(self.a_matrix) > 0)

    def get_convergence_profile(self) -> List:
        """Return convergence profile."""
        return self.residual_values

    def show_convergence_profile(self) -> None:
        """Show plot with convergence profile - normalised residual vector vs iteration."""
        y_es = self.get_convergence_profile()
        x_es = [i for i in range(len(y_es))]
        plt.title(self.name + ' method convergence profile')
        plt.ylabel('Convergence (residual norm)')
        plt.xlabel('Iterations')
        plt.plot(x_es, y_es, 'b--')
        plt.legend(['Total iter = ' + str(self.finished_iter)])
        plt.show()

    @staticmethod
    def compare_convergence_profiles(*args: 'IterativeSolver') -> None:
        """Show plot with multiple convergence profiles."""
        _to_print = []
        _legend = []
        plot_lines = ['-', '--', '*', '^']
        plot_colors = ['b', 'r', 'g', 'y']
        for ind, solver in enumerate(args):
            _y = solver.get_convergence_profile()
            _x = [i for i in range(len(_y))]
            try:
                line_color = plot_colors[ind] + plot_lines[ind]
            except IndexError:
                line_color = ''
            _to_print.append((_x, _y, line_color))
            _legend.append('{} iter = {}'.format(solver.name, solver.finished_iter))
        plt.title('Convergence profiles comparison')
        plt.ylabel('Convergence (residual norm)')
        plt.xlabel('Iterations')
        plt.plot(*[item for sublist in _to_print for item in sublist])
        plt.legend(_legend)
        plt.show()

    @abstractmethod
    def solve(self) -> Tuple[np.matrix, int]:
        """Solve system of linear equations."""
        raise NotImplementedError
