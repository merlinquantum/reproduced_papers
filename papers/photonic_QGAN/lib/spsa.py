import numpy as np


def bernoulli_delta(n_params, p=0.5):
    delta_k = np.random.binomial(1, p, n_params)
    delta_k[delta_k == 0] = -1
    return delta_k


class SPSA:
    """
    SPSA wrapper class used for optimising the generator.

    Matches the original photonic-qgan-main implementation (helpers/spsa.py).

    :param init_params: Flat numpy array of initial parameters to optimise.
    :param grad_fun: Callable returning the pseudo-gradient vector
        ``grad_fun(params, c) -> np.ndarray``.
    :param iter_num: Total number of SPSA iterations (used to set schedule).
    """

    def __init__(self, init_params, grad_fun, iter_num=5000):
        self.params = np.array(init_params, dtype=float)
        self.grad_fun = grad_fun

        self.gamma = 0.101
        self.alpha = 0.602

        self.iter_num = iter_num
        self.k = 0
        self.c = 0.1
        self.A = 0.1 * iter_num

        mag_g0 = np.abs(np.array(self.grad_fun(init_params, self.c)).mean())
        self.a = 0.001 * ((self.A + 1) ** self.alpha) / mag_g0

    def step(self, iter_count=20):
        """Perform a step comprising *iter_count* SPSA iterations.

        :param iter_count: Number of inner iterations for this step.
        :return: Updated parameter array.
        """
        params = self.params.copy()

        for _ in range(iter_count):
            self.k += 1
            ak = self.a / ((self.k + self.A) ** self.alpha)
            ck = self.c / (self.k**self.gamma)
            gk = np.array(self.grad_fun(params, ck))
            params -= ak * gk

        self.params = params
        return params
