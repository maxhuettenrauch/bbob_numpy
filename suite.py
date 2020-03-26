import numpy as np

from experiments.objective_functions.f_base import BaseObjective
from experiments.objective_functions.f_sphere import Sphere as f1
from experiments.objective_functions.f_ellipsoid import Ellipsoid as f2
from experiments.objective_functions.f_rastrigin import Rastrigin as f3
from experiments.objective_functions.f_bueche_rastrigin import BuecheRastrigin as f4
from experiments.objective_functions.f_linear_slope import LinearSlope as f5
from experiments.objective_functions.f_attractive_sector import AttractiveSector as f6
from experiments.objective_functions.f_step_ellipsoid import StepEllipsoid as f7
from experiments.objective_functions.f_rosenbrock import Rosenbrock as f8
from experiments.objective_functions.f_rosenbrock import RosenbrockRotated as f9
from experiments.objective_functions.f_ellipsoid import EllipsoidRotated as f10
from experiments.objective_functions.f_discus import Discus as f11
from experiments.objective_functions.f_bent_cigar import BentCigar as f12
from experiments.objective_functions.f_sharp_ridge import SharpRidge as f13
from experiments.objective_functions.f_different_powers import DifferentPowers as f14
from experiments.objective_functions.f_rastrigin import RastriginRotated as f15
from experiments.objective_functions.f_weierstrass import Weierstrass as f16
from experiments.objective_functions.f_schaffers import Schaffers as f17
from experiments.objective_functions.f_schaffers import Schaffers as f18
from experiments.objective_functions.f_griewank_rosenbrock import GriewankRosenbrock as f19
from experiments.objective_functions.f_schwefel import Schwefel as f20
from experiments.objective_functions.f_gallagher import Gallagher101 as f21
from experiments.objective_functions.f_gallagher import Gallagher21 as f22
from experiments.robotics.hole_reaching_objective import HoleReachingObjective as hro
from experiments.robotics.planar_reaching_objective import ReachingObjective as ro
from experiments.robotics.hole_reaching_objective import HoleReachingEnv as hro_par

from cma.bbobbenchmarks import nfreefunclasses

# TODO: how to make this nicer?
_all_objectives = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, hro, ro]


def get_id_from_name(name: str):
    if name.lower() == 'sphere':
        return 0
    elif name.lower() == 'ellipsoid':
        return 1
    elif name.lower() == 'rastrigin':
        return 2
    elif name.lower() == 'buecherastrigin':
        return 3
    elif name.lower() == 'linearslope':
        return 4
    elif name.lower() == 'attractivesector':
        return 5
    elif name.lower() == 'stepellipsoid':
        return 6
    elif name.lower() == 'rosenbrock':
        return 7
    elif name.lower() == 'rosenbrockrotated':
        return 8
    elif name.lower() == 'ellipsoidrotated':
        return 9
    elif name.lower() == 'discus':
        return 10
    elif name.lower() == 'bentcigar':
        return 11
    elif name.lower() == 'sharpridge':
        return 12
    elif name.lower() == 'differentpowers':
        return 13
    elif name.lower() == 'rastriginrotated':
        return 14
    elif name.lower() == 'weierstrass':
        return 15
    elif name.lower() == 'schaffersf7':
        return 16
    elif name.lower() == 'schaffersf7ill':
        return 17
    elif name.lower() == 'griewankrosenbrock':
        return 18
    elif name.lower() == 'schwefel':
        return 19
    elif name.lower() == 'gallagher101':
        return 20
    elif name.lower() == 'gallagher21':
        return 21
    elif name.lower() == 'katsuura':
        return 22
    elif name.lower() == 'lunacek':
        return 23


class Problem:
    """ wrapper around objective functions for optimization """
    def __init__(self, name: str, dim: int, f_obj: BaseObjective = None, seed=None):
        if f_obj is not None:
            self.f_obj = f_obj(dim)
        else:
            if name.lower() == 'sphere':
                self.f_obj = f1(dim)
            elif name.lower() == 'ellipsoid':
                self.f_obj = f2(dim)
            elif name.lower() == 'rastrigin':
                self.f_obj = f3(dim)
            elif name.lower() == 'buecherastrigin':
                self.f_obj = f4(dim)
            elif name.lower() == 'linearslope':
                self.f_obj = f5(dim)
            elif name.lower() == 'attractivesector':
                self.f_obj = f6(dim)
            elif name.lower() == 'stepellipsoid':
                self.f_obj = f7(dim)
            elif name.lower() == 'rosenbrock':
                self.f_obj = f8(dim)
            elif name.lower() == 'rosenbrockrotated':
                self.f_obj = f9(dim)
            elif name.lower() == 'ellipsoidrotated':
                self.f_obj = f10(dim)
            elif name.lower() == 'discus':
                self.f_obj = f11(dim)
            elif name.lower() == 'bentcigar':
                self.f_obj = f12(dim)
            elif name.lower() == 'sharpridge':
                self.f_obj = f13(dim)
            elif name.lower() == 'differentpowers':
                self.f_obj = f14(dim)
            elif name.lower() == 'rastriginrotated':
                self.f_obj = f15(dim)
            elif name.lower() == 'weierstrass':
                self.f_obj = f16(dim)
            elif name.lower() == 'schaffersf7':
                self.f_obj = f17(dim)
            elif name.lower() == 'schaffersf7ill':
                self.f_obj = f18(dim, alpha=1000)
            elif name.lower() == 'griewankrosenbrock':
                self.f_obj = f19(dim)
            elif name.lower() == 'schwefel':
                self.f_obj = f20(dim)
            elif name.lower() == 'gallagher101':
                self.f_obj = f21(dim)
            elif name.lower() == 'gallagher21':
                self.f_obj = f22(dim)
            elif name.lower() == 'holereach5':
                self.f_obj = hro(num_links=5, learn_goal=True)
            elif name.lower() == 'holereach5_par':
                self.f_obj = hro_par("holereach", 4, learn_goal=True)
            elif name.lower() == 'holereach5_self':
                self.f_obj = hro(num_links=5, allow_self_collision=True)
            elif name.lower() == 'reach5':
                self.f_obj = ro(num_links=5, via_points=({"t": 50, "vp": (1, 1)}, ))
            else:
                raise ValueError('Unknown objective function')
        self.name = name
        self.seed = seed
        self.best_observed_f_value = np.infty

    def __call__(self, x):
        return self.f_obj(x)

    def __str__(self):
        return str(self.f_obj)

    def getfopt(self):
        return self.f_obj.f_opt

    @property
    def allow_parallel(self):
        try:
            return self.f_obj.allow_parallel
        except AttributeError:
            return False

    @property
    def funId(self):
        return self.name + "_" + str(self.seed)

    @property
    def dim(self):
        return self.f_obj.dim

    def f0(self, x):
        """ 0 centered optimal value"""
        f0 = self.f_obj(x) - self.f_obj.f_opt
        if np.min(f0) < self.best_observed_f_value:
            self.best_observed_f_value = np.min(f0)
        return f0

    def final_target_hit(self):
        return True if self.best_observed_f_value < 1e-8 else False


class Suite:
    def __init__(self, suite_options: dict):
        suite_name = suite_options['name']
        dims = suite_options['dim']
        n_instances = suite_options['n_instances']

        self.problems = []
        if suite_name == 'full':
            for n, f in enumerate(_all_objectives):
                for d in dims:
                    for i in range(n_instances):
                        self.problems.append(Problem(f(d), "f{}".format(n+1), i))
        else:
            raise ValueError('suite name not supported')

        self.current_problem = 0
        self.nr_problems = len(self.problems)

    def __iter__(self):
        return self.problems

    def __next__(self):
        try:
            p = self.problems[self.current_problem]
        except IndexError:
            raise StopIteration

        self.current_problem += 1
        return p

    # def next_problem(self):
    #     try:
    #         for p in self:
    #
    #     except StopIteration:
    #         return None


if __name__ == '__main__':
    test_suite_options = {'name': 'full',
                          'dim': [2],
                          'n_instances': 2}
    test_suite = Suite(test_suite_options)

    for p in test_suite:
        print(p.f_obj.f_opt)