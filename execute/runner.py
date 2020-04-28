import numpy as np

from algorithm import ALGORITHMS
from algorithm.base_algorithm import PrivacyPreservingLogistic
from algorithm.huang2015 import Huang2015
from algorithm.nozari2018 import Nozari2018
from algorithm.ppsc import PPSC

n = 10
l = 3000
dim = 28 ** 2
step_size_scale = 1
lamb = 0.5

eval_period = 10

init_network_state = PrivacyPreservingLogistic.gen_init_state(n, dim)

train_data = PrivacyPreservingLogistic.read_data(r'../data/mnist_train.csv')
test_data = PrivacyPreservingLogistic.read_data(r'../data/mnist_test.csv')


def run_ppsc(epsilon):
    alg = PPSC(n=n, epsilon=epsilon, l=l, dim=dim, lamb=lamb, train_data=train_data, test_data=test_data,
               init_network_state=init_network_state, step_size_scale=step_size_scale,
               eval_period=eval_period)
    alg.run()
    alg.save_auc(ALGORITHMS[0] + str(np.log10(epsilon)))


def run_huang2015(epsilon):
    alg = Huang2015(n=n, epsilon=epsilon, l=l, dim=dim, lamb=lamb, train_data=train_data, test_data=test_data,
                    init_network_state=init_network_state, step_size_scale=step_size_scale,
                    eval_period=eval_period)
    alg.run()
    alg.save_auc(ALGORITHMS[1] + str(np.log10(epsilon)))


def run_nozari2018(epsilon):
    alg = Nozari2018(n=n, epsilon=epsilon, l=l, dim=dim, lamb=lamb, train_data=train_data, test_data=test_data,
                     init_network_state=init_network_state, step_size_scale=step_size_scale,
                     eval_period=eval_period)
    alg.run()
    alg.save_auc(ALGORITHMS[2] + str(np.log10(epsilon)))


epsilon_array = [1e-3, 1e-2, 1e-1]
for epsilon in epsilon_array:
    run_ppsc(epsilon)
    run_huang2015(epsilon)
    run_nozari2018(epsilon)
