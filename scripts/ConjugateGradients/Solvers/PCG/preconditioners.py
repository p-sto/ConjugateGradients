"""Contains implementations of pre-conditioners for PCG method."""

from typing import Callable
import numpy as np


def get_preconditioner(name: str) -> Callable:
    """Return pre-conditioner based on name."""
    if name == 'jacobi':
        return jacobi
    raise KeyError('No pre-conditioner for provided name = {}'.format(name))


def jacobi(a_matrix: np.matrix, residual: np.matrix) -> np.matrix:
    """Return vector(np.matrix) obtained by multiplication of inverted a_matrix argument diagonal by residual vector."""
    _to_return = {}  # type: dict
    if 'inverted' not in _to_return:
        # we want to calculate matrix inversion only once...
        _to_return['inverted'] = np.linalg.inv(np.diag(np.diag(a_matrix)))
    return np.matrix(_to_return['inverted'] * residual)
