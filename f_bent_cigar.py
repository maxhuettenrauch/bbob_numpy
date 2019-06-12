import numpy as np
from experiments.objective_functions.f_base import BaseObjective
import logging


logger = logging.getLogger('BentCigar')
logger.warning("not sure if this is correct")


class BentCigar(BaseObjective):
    def __init__(self, d, int_opt=(-5., 5.), beta=0.5):
        super(BentCigar, self).__init__(d, int_opt=int_opt, beta=beta)

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        # FIXME: is this correct?
        z = (self.r @ self.t_asy_beta(self.r @ (x - self.x_opt).T)).T

        return z[:, 0]**2 + 1e6 * np.sum(z[:, 1:]**2, axis=1) + self.f_opt
