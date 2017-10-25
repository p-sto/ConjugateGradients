"""Miscellaneous functions"""

from scripts.ConjugateGradients.Solvers.CG.cg_solver import ConjugateGradientSolver
from scripts.ConjugateGradients.Solvers.PCG.pcg_solver import PreConditionedConjugateGradientSolver
from scripts.ConjugateGradients.Solvers.common import IterativeSolver

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def get_solver(name: str = None) -> IterativeSolver:
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
