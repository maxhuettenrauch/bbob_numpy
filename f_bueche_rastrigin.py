import numpy as np
from experiments.objective_functions.f_base import BaseObjective


class BuecheRastrigin(BaseObjective):
    def __init__(self, d, int_opt=(-5., 5.)):
        super(BuecheRastrigin, self).__init__(d, int_opt)

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        y = self.t_osz(x - self.x_opt)
        s = np.empty_like(y)

        for i in range(self.d):
            if i % 2 == 0:
                s[:, i] = np.where(y[:, i] > 0,
                                   10 * np.power(np.sqrt(10), i / (self.d - 1)),
                                   np.power(np.sqrt(10), i / (self.d - 1)))

        z = s * y

        out = 10 * (self.d - np.sum(np.cos(2 * np.pi * z), axis=1)) + np.linalg.norm(z, axis=1)**2 \
            + 100 * self.f_pen(x) + self.f_opt

        return out
