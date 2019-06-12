import numpy as np
from experiments.objective_functions.f_base import BaseObjective


class Schaffers(BaseObjective):
    def __init__(self, d, int_opt=(-5., 5.), alpha=10, beta=0.5):
        super(Schaffers, self).__init__(d, int_opt=int_opt, alpha=alpha, beta=beta)
        self.mat_fac = self.lambda_alpha @ self.q

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        z = (self.mat_fac @ self.t_asy_beta(self.r @ (x - self.x_opt).T)).T

        s = np.sqrt(z[:, :-1] ** 2 + z[:, 1:] ** 2)

        out = (1 / (self.d - 1) * np.sum(np.sqrt(s) + np.sqrt(s) * np.sin(50 * s ** 0.2) ** 2, axis=1)) ** 2 \
            + 10 * self.f_pen(x) + self.f_opt
        return out
