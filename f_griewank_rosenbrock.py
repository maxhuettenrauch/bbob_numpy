import numpy as np
from experiments.objective_functions.f_base import BaseObjective


class GriewankRosenbrock(BaseObjective):
    def __init__(self, d, int_opt=(-5., 5.)):
        super(GriewankRosenbrock, self).__init__(d, int_opt=int_opt)
        self.c = np.maximum(1, np.sqrt(self.d) / 8)

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        z = (self.c * self.r @ x.T + 1/2).T
        a = z[:, :-1]**2 - z[:, 1:]
        b = z[:, :-1] - 1

        s = 100 * a**2 + b**2

        out = 10 / (self.d - 1) * np.sum(s / 4000 - np.cos(s), axis=1) + 10 + self.f_opt

        return out
