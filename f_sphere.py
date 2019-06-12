import numpy as np
from experiments.objective_functions.f_base import BaseObjective


class Sphere(BaseObjective):
    def __init__(self, d, int_opt=(-5., 5.)):
        super(Sphere, self).__init__(d, int_opt)

    @staticmethod
    def id():
        return "f1"

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        z = x - self.x_opt
        return np.linalg.norm(z, axis=1)**2 + self.f_opt


if __name__ == "__main__":
    test = Sphere(2)
    test(np.zeros(2))
