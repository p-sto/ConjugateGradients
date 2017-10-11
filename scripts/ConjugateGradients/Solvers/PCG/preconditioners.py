"""Contains implementations of pre-conditioners for PCG method."""

import numpy as np


def jacobi(a_matrix: np.matrix, residual: np.matrix) -> np.matrix:
    """Return vector(np.matrix) obtained by multiplication of inverted a_matrix argument diagonal by residual vector."""
    _ = {}
    if 'inverted' not in _:
        # we want to calculate matrix inversion only once...
        _['inverted'] = np.linalg.inv(np.diag(np.diag(a_matrix)))
    return np.matrix(_['inverted'] * residual)
