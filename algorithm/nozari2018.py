import copy
import random

from algorithm.base_algorithm import PrivacyPreservingLogistic
import functools
from scipy import integrate
from scipy.misc import derivative
from scipy.special import zeta
import numpy as np


class Nozari2018(PrivacyPreservingLogistic):
    def __init__(self, train_data, test_data, n, epsilon, l, dim, lamb, init_network_state, step_size_scale, eval_period,
                 gamma=2e3, p=0.55):
        super(Nozari2018, self).__init__(n=n,
                                         epsilon=epsilon,
                                         l=l,
                                         dim=dim,
                                         lamb=lamb,
                                         train_data=train_data,
                                         test_data=test_data,
                                         init_network_state=init_network_state,
                                         step_size_scale=step_size_scale,
                                         eval_period=eval_period)

        self.gamma = gamma
        self.p = p

        self._init_param()

        self._init_phi()

        self.concat_noise_grad = None

    def _init_param(self):
        # compute q
        zeta_val = np.sqrt(self.epsilon * self.gamma)
        self.q = self.p + self.inverse_zeta(zeta_val) / 2

    def _init_phi(self):
        """
        the noise function
        :return:
        """
        self.phi_array = []

        def func_phi(x, eta_array, basis):
            """

            :param x: ndarray whose shape is (dim, 1)
            :return:
            """
            func_val = 0
            for i in range(len(basis)):
                func_val += eta_array[i] * basis[i](x)
            return func_val

        # init basis
        basis = self.gen_multivariable_taylor1(self.dim)
        random.shuffle(basis)

        for i in range(self.n):
            # compute b_k
            b_k_seq = [self.gamma / (k + 1) ** self.p for k in range(len(basis))]

            # generate eta
            eta_array = []
            for b_k in b_k_seq:
                eta = self.gen_laplace_random_variable(1, 0, b_k)[0]
                eta_array.append(eta)

            # compute phi, which is a function list
            phi = functools.partial(func_phi, eta_array=eta_array, basis=basis)
            self.phi_array.append(phi)

    def _iterate(self):
        # optimization
        step_size = self._step_size()
        gradient = self._gradient()
        noise_gradient = self._noise_gradient()
        gradient += noise_gradient
        self.network_state = self.network_state + step_size * gradient

        # project
        self._project()

    def _noise_gradient(self, fast_mode=True):
        """
        compute noise gradient
        :return:
        """
        noise_gradient_array = []
        if fast_mode:
            if self.concat_noise_grad is not None:
                return self.concat_noise_grad
            for i in range(self.n):
                phi = self.phi_array[i]
                node_grad = np.zeros((self.dim, 1))

                for j in range(self.dim):
                    try:
                        eta = phi.keywords['eta_array'][j]
                        node_idx = phi.keywords['basis'][j].keywords['i']
                        node_grad[int(node_idx), 0] = eta
                    except:
                        pass

                noise_gradient_array.append(node_grad)
        else:
            for i in range(self.n):
                node_state = self.network_state[i * self.dim:(i + 1) * self.dim]
                noise_gradient = self.gradient(self.phi_array[i], node_state)
                noise_gradient_array.append(noise_gradient)

        concat_noise_grad = noise_gradient_array.pop(0)
        for noise_gradient in noise_gradient_array:
            concat_noise_grad = np.concatenate((concat_noise_grad, noise_gradient))

        self.concat_noise_grad = concat_noise_grad

        return self.concat_noise_grad

    @staticmethod
    def inverse_zeta(val):
        precision = 1e-5
        x = 1
        while True:
            x += precision
            if zeta(x) < val:
                return x

    @staticmethod
    def func_inner_dot(f, g, lower=-1, upper=1):
        def func_product(x, f, g):
            return f(x) * g(x)

        product = integrate.quad(functools.partial(func_product, f=f, g=g), lower, upper)[0]
        return product

    @staticmethod
    def multiv_func_inner_dot(f, g, lower, upper):
        def func_product(x, f, g):
            return f(x) * g(x)

        product = integrate.nquad(functools.partial(func_product, f=f, g=g), lower, upper)[0]
        return product

    @staticmethod
    def func_square(f, lower=-1, upper=1):
        return Nozari2018.func_inner_dot(f, f, lower, upper)

    @staticmethod
    def gram_schmidt(input_list):
        """
        a set of independent functions
        :return: a basis, i.e., a set orthogonal functions
        """
        def create_func(x, input_list, coef_array, basis, i):
            func_val = input_list[i](x)
            for j in range(i):
                func_val += coef_array[j] * basis[j](x)
            return func_val

        basis = []
        for i in range(len(input_list)):
            coef_array = []
            for j in range(i):
                coef = -Nozari2018.func_inner_dot(input_list[i], basis[j]) / Nozari2018.func_inner_dot(basis[j], basis[j])
                coef_array.append(coef)

            basis.append(functools.partial(create_func, input_list=input_list, coef_array=coef_array, basis=basis, i=i))

        return basis

    @staticmethod
    def multiv_gram_schmidt(input_list, dim):
        """
        a set of independent functions
        :return: a basis, i.e., a set orthogonal functions
        """

        def create_func(x, input_list, coef_array, basis, i):
            func_val = input_list[i](x)
            for j in range(i):
                func_val += coef_array[j] * basis[j](x)
            return func_val

        lower = [-1] * dim
        upper = [1] * dim

        basis = []
        for i in range(len(input_list)):
            coef_array = []
            for j in range(i):
                coef = -Nozari2018.multiv_func_inner_dot(input_list[i], basis[j], lower, upper)\
                       /Nozari2018.func_inner_dot(basis[j], basis[j], lower, upper)
                coef_array.append(coef)

            basis.append(functools.partial(create_func, input_list=input_list, coef_array=coef_array, basis=basis, i=i))

        return basis

    @staticmethod
    def gen_taylor_series(upper_order):
        def create_func(x, i):
            return x ** i

        taylor_series = []
        for i in range(upper_order):
            taylor_series.append(functools.partial(create_func, i=i))
        return taylor_series

    @staticmethod
    def gen_basis_from_taylor(upper_order):
        taylor = Nozari2018.gen_taylor_series(upper_order)
        basis = Nozari2018.gram_schmidt(taylor)
        return basis

    @staticmethod
    def derivative(func, x):
        deri_val = derivative(func, x, dx=1e-6)
        return deri_val

    @staticmethod
    def gradient(func, x):
        delta = 1e-6
        grad = []
        for i in range(len(x)):
            prev_x = copy.deepcopy(x)
            next_x = copy.deepcopy(x)
            prev_x[i] -= delta
            next_x[i] += delta
            grad.append((func(next_x) - func(prev_x)) / (2 * delta))
        return grad

    @staticmethod
    def gen_multivariable_taylor1(dim):
        """

        :return: func_list: [1, x1, x2, ...]
        """

        def create_1(x):
            return 1.0

        def create_xi(x, i):
            return x[i]

        taylor = [create_1]

        for i in range(dim):
            taylor.append(functools.partial(create_xi, i=i))

        return taylor

    @staticmethod
    def gen_multivariable_taylor2(dim):
        """

        :return: func_list: [1, x1, x2, ..., x1^2, x2^2, ..., x1x2, x1x3, ...]
        """
        def create_1(*x):
            return 1.0

        def create_xi(*x, i):
            return x[i]

        def create_xi_square(*x, i):
            return x[i] ** 2

        def create_crossing(*x, i, j):
            return x[i] * x[j]

        taylor = [create_1]

        for i in range(dim):
            taylor.append(functools.partial(create_xi, i=i))

        for i in range(dim):
            taylor.append(functools.partial(create_xi_square, i=i))

        for i in range(dim - 1):
            for j in range(i, dim):
                taylor.append(functools.partial(create_crossing, i=i, j=j))

        return taylor

    @staticmethod
    def gen_multivariable_taylor2_no_crossing(dim):
        """

        :return: func_list: [1, x1, x2, ..., x1^2, x2^2, ...]
        """

        def create_xi(*x):
            return x[i]

        def create_xi_square(*x, i):
            return x[i] ** 2

        taylor = [lambda x: 1.0]

        for i in range(dim):
            taylor.append(functools.partial(create_xi, i=i))

        for i in range(dim):
            taylor.append(functools.partial(create_xi_square, i=i))

        return taylor

    @staticmethod
    def gen_basis_from_taylor_multivariable(dim):
        taylor = Nozari2018.gen_multivariable_taylor2_no_crossing(dim)
        basis = Nozari2018.multiv_gram_schmidt(taylor, dim)
        return basis
