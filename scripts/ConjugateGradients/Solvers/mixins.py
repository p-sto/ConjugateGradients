"""Mixins for solvers."""


class Convergence:
    """Mixin to obtain convergence profile for solver."""

    def get_convergence_profile(self):
        """Returns convergence profile."""
        # print 'Residual norm:', np.linalg.norm(r)
        pass
