"""Miscellaneous functions"""
from typing import List, Tuple, Callable

from scripts.ConjugateGradients.Solvers.CG.cg_solver import ConjugateGradientSolver
from scripts.ConjugateGradients.Solvers.PCG.pcg_solver import PreConditionedConjugateGradientSolver

import numpy as np
from scipy.sparse.csr import csr_matrix
import matplotlib
import matplotlib.pyplot as plt


class CSRMatrix:
    """Wrap over standard csr_matrix scipy object for easier data manipulation."""

    def __init__(self, m_matrix: np.matrix) -> None:
        self._m_matrix = csr_matrix(m_matrix)

    @property
    def shape(self) -> Tuple[int, int]:
        """Return tuple with shape."""
        return self._m_matrix.shape

    @property
    def rows_i(self) -> List:
        """Return list containing matrix rows indexes."""
        return self._m_matrix.indptr

    @property
    def column_j(self) -> List:
        """Return list containing matrix columns indexes."""
        return self._m_matrix.indices

    @property
    def values(self) -> List:
        """Return list containing matrix values."""
        return list(self._m_matrix.data)

    @property
    def nnz(self) -> int:
        """Return number of non-zero elements."""
        return self._m_matrix.nnz


def get_solver(name: str = None) -> Callable:
    """Return solver based on name."""
    if not name or name == 'CG':
        return ConjugateGradientSolver
    if name == 'PCG':
        return PreConditionedConjugateGradientSolver
    raise NotImplementedError('Provided name for solver is wrong.')


def view_matrix(mat: np.matrix) -> None:
    """Print matrix graphical representation."""
    masked = np.ma.masked_where(mat == -1, mat)
    cmap = matplotlib.cm.viridis    # pylint: disable=no-member
    cmap.set_under(color='white')
    plt.imshow(masked, interpolation='nearest', cmap=cmap, vmin=0.0000001)
    plt.colorbar()
    plt.title('Matrix')
    plt.show()


def save_csr_matrix_to_file(matrix: np.matrix, filename: str):
    """Save CSR matrix to txt file"""
    _matrix = CSRMatrix(matrix)
    if _matrix.shape[0] != _matrix.shape[1]:
        raise TypeError('Sorry - only size x size matrices!')
    with open(filename, 'w+') as fil:
        # save line with number of elements, matrix size and rows number
        fil.write('{} {} {}\n'.format(_matrix.shape[0], _matrix.nnz, len(_matrix.rows_i)))
        if len(_matrix.rows_i) < len(_matrix.column_j):
            for ind in range(len(_matrix.column_j)):
                if ind < len(_matrix.rows_i):
                    fil.write('{} {} {}\n'.format(_matrix.values[ind], _matrix.column_j[ind], _matrix.rows_i[ind]))
                else:
                    fil.write('{} {}\n'.format(_matrix.values[ind], _matrix.column_j[ind]))
        else:
            for ind in range(len(_matrix.rows_i)):
                if ind < len(_matrix.column_j):
                    fil.write('{} {} {}\n'.format(_matrix.values[ind], _matrix.column_j[ind], _matrix.rows_i[ind]))
                else:
                    fil.write('{} {} {}\n'.format('-', '-', _matrix.rows_i[ind]))
