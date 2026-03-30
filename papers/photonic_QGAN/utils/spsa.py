import numpy as np


class SPSA:
    """
    SPSA wrapper class used for optimising the models.

    :param init_params: Values of initial parameters over which the optimization should be performed.
    :param grad_fun: Method returning the pseudo-gradient vector w.r.t. the parameters.
    :param iter_num: The number of SPSA iterations.
    """

    def __init__(self, init_params, grad_fun, iter_num=5000):
        self.params = init_params
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
        """Perform a step of optimization consisting of a given number of iterations.

        :param iter_count: The duration of the step aka the number of iterations is comprises.
        :return: the updated parameters
        """
        params = self.params.copy()

        for _ in range(iter_count):
            self.k += 1
            ak = self.a / ((self.k + self.A) ** (self.alpha))
            ck = self.c / (self.k ** (self.gamma))
            gk = np.array(self.grad_fun(params, ck))
            params -= ak * gk

        self.params = params
        return params
