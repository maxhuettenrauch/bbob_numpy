import numpy as np
from experiments.objective_functions.f_base import BaseObjective


class Rastrigin(BaseObjective):
    def __init__(self, d, int_opt=(-5., 5.), alpha=10, beta=0.2):
        super(Rastrigin, self).__init__(d, int_opt, alpha=alpha, beta=beta)
        self.mat_fac = self.lambda_alpha

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        # TODO: maybe flip data dimensions?
        z = (self.mat_fac @ self.t_asy_beta(self.t_osz(x - self.x_opt).T)).T

        out = 10 * (self.d - np.sum(np.cos(2 * np.pi * z), axis=1)) + np.linalg.norm(z, axis=1)**2 + self.f_opt

        return out


class RastriginRotated(BaseObjective):
    def __init__(self, d, int_opt=(-5., 5.), alpha=10, beta=0.2):
        super(RastriginRotated, self).__init__(d, int_opt, alpha=alpha, beta=beta)
        self.mat_fac = self.r @ self.lambda_alpha @ self.q

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        # TODO: maybe flip data dimensions?
        z = (self.mat_fac @ self.t_asy_beta(self.t_osz(self.r @ (x - self.x_opt).T))).T

        out = 10 * (self.d - np.sum(np.cos(2 * np.pi * z), axis=1)) + np.linalg.norm(z, axis=1)**2 + self.f_opt

        return out
