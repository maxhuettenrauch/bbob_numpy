import numpy as np
from experiments.objective_functions.f_base import BaseObjective


class Schwefel(BaseObjective):
    def __init__(self, d, val_opt=4.2096874633 / 2, alpha=10):
        super(Schwefel, self).__init__(d, val_opt=val_opt, alpha=alpha)

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        x_hat = 2 * self.one_pm * x

        z_hat = np.zeros_like(x)
        z_hat[:, 0] = x_hat[:, 0]
        z_hat[:, 1:] = x_hat[:, 1:] + 0.25 * (x_hat[:, :-1] - 2 * np.abs(self.x_opt[:, :-1]))

        z = 100 * (self.lambda_alpha @ (z_hat - 2 * np.abs(self.x_opt)).T + 2 * np.abs(self.x_opt).T).T

        return -1 / (100 * self.d) * np.sum(z * np.sin(np.sqrt(np.abs(z))), axis=1) + 4.189828872724339 \
            + 100 * self.f_pen(z / 100) + self.f_opt
