"""BFGS with a faster solve for positive-definite Hessians.

ASE diagonalises the full Hessian on every BFGS step so it can take the
absolute value of negative eigenvalues.  A positive-definite Hessian needs no
such correction: solving ``H * step = -gradient`` gives the same BFGS step and
is substantially cheaper for the large cells used in NEP89 pre-relaxation.
The original ASE step remains the fallback when Cholesky cannot factor ``H``.
"""

from __future__ import annotations

import numpy as np
from ase.optimize import BFGS as ASEBFGS
from scipy.linalg import cho_factor, cho_solve


class BFGS(ASEBFGS):
    """Use Cholesky when possible while retaining ASE's ``BFGS:`` log lines."""

    def prepare_step(self, pos, gradient):
        # Keep ASE's Hessian update, force sign and step limiting unchanged.
        pos = pos.ravel()
        gradient = gradient.ravel()
        self.update(pos, -gradient, self.pos0, self.forces0)
        self.pos0 = pos
        self.forces0 = -gradient.copy()

        try:
            # NumPy's eigh (ASE's path) also reads the lower triangle.
            factor = cho_factor(self.state.hessian, lower=True, check_finite=False)
            dpos = -cho_solve(factor, gradient, check_finite=False)
            if not np.all(np.isfinite(dpos)):
                raise np.linalg.LinAlgError("non-finite Cholesky step")
        except np.linalg.LinAlgError:
            # ASE's eigenvalue rule is needed for an indefinite Hessian.
            dpos = self.state.compute_step(gradient)

        steplengths = self.optimizable.gradient_norm(dpos)
        return dpos, steplengths


# The descriptive import name keeps call sites clear.  ASE uses the class name
# in its optimiser log, where existing parsers expect exactly ``BFGS:``.
PositiveDefiniteBFGS = BFGS
