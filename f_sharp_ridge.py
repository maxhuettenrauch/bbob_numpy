import numpy as np
from experiments.objective_functions.f_base import BaseObjective
import logging


logger = logging.getLogger('SharpRidge')


class SharpRidge(BaseObjective):
    def __init__(self, d, int_opt=(-5., 5.), alpha=10):
        logger.warning("not sure if this is correct")
        super(SharpRidge, self).__init__(d, int_opt=int_opt, alpha=alpha)

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        z = (self.q @ self.lambda_alpha @ self.r @ (x - self.x_opt).T).T
        out = z[:, 0] ** 2 + 100 * np.sqrt(np.sum(z[:, 1:] ** 2, axis=1)) + self.f_opt

        return out
