import numpy as np
from experiments.objective_functions.f_base import BaseObjective
import logging


logger = logging.getLogger("Gallagher")


# FIXME: Something's probably wrong in here
class Gallagher21(BaseObjective):
    def __init__(self, d):
        logger.warning("Check implementation for errors")
        i = 20
        int_loc_opt = (-4.9, 4.9)
        alpha_1 = 1000**2
        int_opt = (-3.92, 3.92)
        alpha_set = np.power(1000, 2 * np.arange(i)/(i-1))
        alpha = np.hstack([alpha_1, np.random.choice(alpha_set, i, replace=False)])
        super(Gallagher21, self).__init__(d, int_opt=int_opt, alpha=alpha)
        self.w = np.hstack([10, 1.1 + 8 * np.arange(i)/(i-1)])
        self.C = self.lambda_alpha / self.alpha[:, None, None]**0.25
        self.mat_fac = self.r.T @ self.C @ self.r

        self.y = np.vstack([self.x_opt, np.random.uniform(int_loc_opt[0], int_loc_opt[1], size=(i, d))])

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        z = x[None, :, None, :] - self.y[:, None, None, :]

        power = -1 / (2 * self.d) * np.squeeze(z @ self.mat_fac[:, None, :, :] @ np.transpose(z, [0, 1, 3, 2]),
                                               axis=(2, 3))
        max_i = np.max(self.w[:, None] * np.exp(power), axis=0)

        out = self.t_osz(10 - max_i)**2 + self.f_pen(x) + self.f_opt
        return out


class Gallagher101(BaseObjective):
    def __init__(self, d):
        logger.warning("Check implementation for errors")
        i = 100
        int_loc_opt = (-5, 5)
        alpha_1 = 1000
        int_opt = (-4., 4.)
        alpha_set = np.power(1000, 2 * np.arange(i)/(i-1))

        alpha = np.hstack([alpha_1, np.random.choice(alpha_set, i, replace=False)])
        super(Gallagher101, self).__init__(d, int_opt=int_opt, alpha=alpha)
        self.w = np.hstack([10, 1.1 + 8 * np.arange(i)/(i-1)])
        self.C = self.lambda_alpha / self.alpha[:, None, None]**0.25
        self.mat_fac = self.r.T @ self.C @ self.r

        self.y = np.vstack([self.x_opt, np.random.uniform(int_loc_opt[0], int_loc_opt[1], size=(i, d))])

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        z = x[None, :, None, :] - self.y[:, None, None, :]

        power = -1 / (2 * self.d) * np.squeeze(z @ self.mat_fac[:, None, :, :] @ np.transpose(z, [0, 1, 3, 2]),
                                               axis=(2, 3))
        max_i = np.max(self.w[:, None] * np.exp(power), axis=0)

        out = self.t_osz(10 - max_i)**2 + self.f_pen(x) + self.f_opt
        return out
