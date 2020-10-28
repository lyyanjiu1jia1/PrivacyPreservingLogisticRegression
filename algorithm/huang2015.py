import numpy as np

from algorithm.base_algorithm import PrivacyPreservingLogistic


class Huang2015(PrivacyPreservingLogistic):
    def __init__(self, train_data, test_data, n, epsilon, l, dim, lamb, init_network_state, step_size_scale,
                 eval_period, metric,
                 c2=1):
        super(Huang2015, self).__init__(n=n,
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

        self.algorithm_name = 'huang2015'

        self.c2 = c2

        self._gen_param()

    def _iterate(self):
        # obfuscate
        noise_lambda = self._laplace_strength()
        noise = self.gen_laplace_random_variable((self.n * self.dim, 1), 0, noise_lambda)
        self.network_state += noise

        # standard average
        self.network_state = self.A_tilde.dot(self.network_state)

        # optimization
        step_size = self._step_size()
        gradient = self._gradient()
        self.network_state = self.network_state + step_size * gradient

        # project
        self._project()

    def _gen_param(self):
        self.c = np.random.randint(0, 100)
        self.q = np.random.uniform(0.5, 1)
        self.p = np.random.uniform(self.q, 1)

    def _step_size(self):
        return self.c * self.q ** (self.iteration - 1)

    def _laplace_strength(self):
        return 2 * self.c2 * np.sqrt(self.n) * self.c * self.p ** self.iteration / (self.epsilon * (self.p - self.q))
