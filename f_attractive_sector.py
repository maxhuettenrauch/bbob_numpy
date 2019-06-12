import numpy as np
from experiments.objective_functions.f_base import BaseObjective


class AttractiveSector(BaseObjective):
    def __init__(self, d, int_opt=(-5., 5.), alpha=10):
        super(AttractiveSector, self).__init__(d, int_opt=int_opt, alpha=alpha)
        self.mat_fac = self.q @ self.lambda_alpha @ self.r

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        z = (self.mat_fac @ (x - self.x_opt).T).T
        s = np.where(z * self.x_opt > 0, 100, 1)
        out = self.t_osz(np.sum((s * z)**2, axis=1))**0.9 + self.f_opt
        return out
