import numpy as np
from experiments.objective_functions.f_base import BaseObjective


class Ellipsoid(BaseObjective):
    def __init__(self, d, int_opt=(-5., 5.)):
        super(Ellipsoid, self).__init__(d, int_opt)
        self.c = np.power(1e6, self.i / (self.d - 1))

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        z = self.t_osz(x - self.x_opt)

        out = np.sum(self.c * z**2, axis=1) + self.f_opt
        return out


class EllipsoidRotated(BaseObjective):
    def __init__(self, d, int_opt=(-5., 5.)):
        super(EllipsoidRotated, self).__init__(d, int_opt)
        self.c = np.power(1e6, self.i / (self.d - 1))

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        z = self.t_osz(self.r @ (x - self.x_opt).T).T
        return np.sum(self.c * z**2, axis=1) + self.f_opt


class EllipsoidRaw(BaseObjective):
    def __init__(self, d, int_opt=(-5., 5.)):
        super(EllipsoidRaw, self).__init__(d, int_opt)
        self.c = np.power(1e6, self.i / (self.d - 1))
        self.x_opt = np.zeros(shape=(1, d))
        self.f_opt = 0

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        out = np.sum(self.c * x**2, axis=1) + self.f_opt
        return out
