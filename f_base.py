import numpy as np
import scipy.stats as scistats


np.seterr(divide='ignore', invalid='ignore')


class BaseObjective:
    def __init__(self, d, int_opt=None, val_opt=None, alpha=None, beta=None):
        self.d = d
        self.alpha = alpha
        self.beta = beta
        # check if optimal parameter is in interval...
        if int_opt is not None:
            self.x_opt = np.random.uniform(int_opt[0], int_opt[1], size=(1, d))
        # ... or based on a single value
        elif val_opt is not None:
            self.one_pm = np.where(np.random.rand(1, d) > 0.5, 1, -1)
            self.x_opt = val_opt * self.one_pm
        else:
            raise ValueError("Optimal value or interval has to be defined")
        self.f_opt = np.round(np.clip(scistats.cauchy.rvs(loc=0, scale=100, size=1)[0], -1000, 1000), decimals=2)
        self.i = np.arange(self.d)
        self._lambda_alpha = None
        self._q = None
        self._r = None

    def __call__(self, x):
        return self.evaluate_full(x)

    def evaluate_full(self, x):
        raise NotImplementedError("Subclasses should implement this!")

    # TODO: property probably unnecessary
    @property
    def q(self):
        if self._q is None:
            a = np.random.randn(self.d, self.d)
            # TODO: correct way of doing Gram Schmidt ortho-normalization?
            q, _ = np.linalg.qr(a)
            self._q = q
        return self._q

    @property
    def r(self):
        if self._r is None:
            a = np.random.randn(self.d, self.d)
            # TODO: correct way of doing Gram Schmidt ortho-normalization?
            r, _ = np.linalg.qr(a)
            self._r = r
        return self._r

    @property
    def lambda_alpha(self):
        if self._lambda_alpha is None:
            if isinstance(self.alpha, int):
                lambda_ii = np.power(self.alpha,  1/2 * self.i / (self.d - 1))
                self._lambda_alpha = np.diag(lambda_ii)
            else:
                lambda_ii = np.power(self.alpha[:, None],  1/2 * self.i[None, :] / (self.d - 1))
                self._lambda_alpha = np.stack([np.diag(l_ii) for l_ii in lambda_ii])
        return self._lambda_alpha

    @staticmethod
    def f_pen(x):
        return np.sum(np.maximum(0, np.abs(x) - 5), axis=1)

    def t_asy_beta(self, x):
        exp = np.power(x, 1 + self.beta * self.i[:, None] / (self.d - 1) * np.sqrt(x))
        return np.where(x > 0, exp, x)

    def t_osz(self, x):
        x_hat = np.where(x != 0, np.log(np.abs(x)), 0)
        c_1 = np.where(x > 0, 10, 5.5)
        c_2 = np.where(x > 0, 7.9, 3.1)
        return np.sign(x) * np.exp(x_hat + 0.049 * (np.sin(c_1 * x_hat) + np.sin(c_2 * x_hat)))
