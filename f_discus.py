import numpy as np
from experiments.objective_functions.f_base import BaseObjective


class Discus(BaseObjective):
    def __init__(self, d, int_opt=(-5., 5.)):
        super(Discus, self).__init__(d, int_opt)

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        z = self.t_osz(self.r @ (x - self.x_opt).T).T

        return 1.e6 * z[:, 0]**2 + np.sum(z[:, 1:]**2, axis=1) + self.f_opt
