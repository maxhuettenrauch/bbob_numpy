import numpy as np
from experiments.objective_functions.f_base import BaseObjective


class StepEllipsoid(BaseObjective):
    def __init__(self, d, int_opt=(-5., 5.), alpha=10):
        super(StepEllipsoid, self).__init__(d, int_opt=int_opt, alpha=alpha)
        self.mat_fac = self.lambda_alpha @ self.r

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        z_hat = (self.mat_fac @ (x - self.x_opt).T)
        z_tilde = np.where(np.abs(z_hat) > 0.5, np.floor(0.5 + z_hat), np.floor(0.5 + 10 * z_hat) / 10)
        z = (self.q @ z_tilde).T

        out = 0.1 * np.maximum(np.abs(z_hat[0, :]) / 1e4,
                               np.sum(np.power(100, self.i / (self.d - 1)) * z**2, axis=1)
                               ) + self.f_pen(x) + self.f_opt
        return out
