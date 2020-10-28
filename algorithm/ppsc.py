import random

import numpy as np
from scipy import integrate

from algorithm.base_algorithm import PrivacyPreservingLogistic


class PPSC(PrivacyPreservingLogistic):
    def __init__(self, train_data, test_data, n, epsilon, l, dim, lamb, init_network_state, step_size_scale,
                 eval_period, metric,
                 mu=1, delta=1e-6, p0=1e-3, xi_m=0.2, g_dagger=1, r_presicion=1e-1, lambda_a=2, p1=0.95, v=1e-1,
                 p2=0.95, q=3, n_max=4, r_dagger=0.5, lambda_ppsc=0.7639):
        super(PPSC, self).__init__(n=n,
                                   epsilon=epsilon,
                                   l=l,
                                   dim=dim,
                                   lamb=lamb,
                                   train_data=train_data,
                                   test_data=test_data,
                                   init_network_state=init_network_state,
                                   step_size_scale=step_size_scale,
                                   eval_period=eval_period,
                                   metric=metric)

        self.algorithm_name = 'ppsc'

        self.mu = mu
        self.delta = delta
        self.p0 = p0
        self.xi_m = xi_m
        self.g_dagger = g_dagger
        self.r_precision = r_presicion
        self.lambda_a = lambda_a
        self.p1 = p1
        self.v = v

        self.p2 = p2
        self.q = q
        self.n_max = n_max
        self.r_dagger = r_dagger
        self.lambda_ppsc = lambda_ppsc

        self.all_edges = [[(0, 1), (1, 2)],
                          [(3, 4), (4, 5)],
                          [(6, 7), (7, 8), (8, 9)]]            # three connected components

        self._compute_delta_sharp()
        self._compute_stsigma()
        self._gen_A_power()

    def _iterate(self):
        # ppsc gossip
        self._ppsc_gossip()

        # average
        self._standard_average()

        # optimization
        step_size = self._step_size()
        gradient = self._gradient()
        self.network_state = self.A_tilde.dot(self.network_state) + step_size * gradient

        # project
        self._project()

    # def _compute_stsigma(self):
    #     """
    #     Compute S, T, sigma_gamma
    #     :return:
    #     """
    #     # S
    #     self.s = (np.log(self.p0) - np.log(self.n)) / np.log(1 - self.xi_m)
    #
    #     # sigma
    #     self.r = self._func_r(epsilon=self.epsilon / self.l,
    #                           delta=self.delta_sharp)
    #     self.sigma = self.n * self.mu * self.g_dagger * self.r / np.sqrt(self.lambda_a)
    #
    #     # T
    #     self.phi_dagger = np.sqrt(self.dim)
    #     D, V = np.linalg.eig(self.A)
    #     D.sort()
    #     sigma_n_minus_one = D[-2]
    #     self.alpha_l = 1 / (self.l + 1)
    #     self.t = (np.log(self.p1 * np.sqrt(self.v) * self.alpha_l ** 2) - np.log(self.phi_dagger ** 2 + 2 * self.s ** 2 * self.sigma ** 2)) / np.log(sigma_n_minus_one)
    #
    #     # round
    #     self.s = int(self.s)
    #     self.t = int(self.t)

    def _compute_stsigma(self):
        """
        Compute S, T, sigma_gamma, for multi-gossip
        :return:
        """
        # S
        self.s = (np.log(1 - np.power(self.p2, 1 / (self.q * self.l))) - np.log(self.n_max)) / \
                 np.log(1 - (1 + self.r_dagger) / self.n_max)

        # sigma
        self.r = self._func_r(epsilon=self.epsilon / self.l,
                              delta=self.delta_sharp)
        self.sigma = self.n * self.mu * self.g_dagger * self.r / self.lambda_ppsc

        # T
        self.phi_dagger = np.sqrt(self.dim)
        D, V = np.linalg.eig(self.A)
        D.sort()
        sigma_n_minus_one = D[-2]
        self.alpha_l = 1 / (self.l + 1)
        self.t = (np.log((1 - np.power(self.p1, 1 / self.l)) * self.v * self.alpha_l ** 4) -
                  np.log(self.n * np.square(self.phi_dagger) + 2 * np.square(self.q) * np.square(self.s) * np.square(self.sigma)))
        self.t /= np.log(sigma_n_minus_one)

        # round
        self.s = int(self.s)
        self.t = int(self.t)

    @staticmethod
    def func_q(w):
        val = 1 / np.sqrt(2 * np.pi) * integrate.quad(lambda v: np.exp(-v ** 2 / 2), w, np.inf)[0]
        return val

    def _inverse_func_q(self, delta):
        if delta == 0.5:
            return 0
        w = 0
        if delta > 0.5:
            while True:
                w -= self.r_precision
                func_val = self.func_q(w)
                if func_val > delta:
                    return w
        else:
            while True:
                w += self.r_precision
                func_val = self.func_q(w)
                if func_val < delta:
                    return w

    def _func_r(self, epsilon, delta):
        q_inverse = self._inverse_func_q(delta)
        r = (q_inverse + np.sqrt(q_inverse ** 2 + 2 * epsilon)) / (2 * epsilon)
        return r

    def _compute_delta_sharp(self):
        self.delta_sharp = (self.delta + np.exp(self.epsilon)) ** (1 / self.l) - np.exp(self.epsilon / self.l)

    def _ppsc_gossip(self):
        """
        Suppose n is even
        :return:
        """
        for i in range(self.s):
            for edges in self.all_edges:
                # for each connected component
                selected_edge = random.choice(edges)
                self._ppsc_gossip_edge(selected_edge)

    def _ppsc_gossip_edge(self, edge):
        noise = self.gen_normal_random_variable((self.dim, 1), 0, self.sigma)
        head = edge[0]
        tail = edge[1]
        self.network_state[tail * self.dim:(tail + 1) * self.dim] +=\
            self.network_state[head * self.dim:(head + 1) * self.dim] - noise
        self.network_state[head * self.dim:(head + 1) * self.dim] = noise


    def _standard_average(self):
        self.network_state = self.A_power.dot(self.network_state)

    def _gen_A_power(self):
        self.A_power = np.kron(np.linalg.matrix_power(self.A, self.t), np.identity(self.dim))
