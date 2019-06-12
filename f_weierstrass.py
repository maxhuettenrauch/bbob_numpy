import numpy as np
from experiments.objective_functions.f_base import BaseObjective


class Weierstrass(BaseObjective):
    def __init__(self, d, int_opt=(-5., 5.), alpha=0.01):
        super(Weierstrass, self).__init__(d, int_opt=int_opt, alpha=alpha)
        self.mat_fac = self.r @ self.lambda_alpha @ self.q
        self.k = np.arange(12)
        self.f_0 = np.sum(1 / (2 ** self.k) * np.cos(2 * np.pi * 3**self.k * 1/2))  # maybe remove 2 * 1/2

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        z = (self.mat_fac @ self.t_osz(self.r @ (x - self.x_opt).T)).T

        sum_k = np.zeros_like(z)
        for k in self.k:
            sum_k += 1 / 2 ** k * np.cos(2 * np.pi * 3 ** k * (z + 1 / 2))

        out = 10 * (1 / self.d * np.sum(sum_k, axis=1) - self.f_0) ** 3 + 10/self.d * self.f_pen(x) + self.f_opt
        return out
