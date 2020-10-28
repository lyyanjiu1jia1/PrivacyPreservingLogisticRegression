import copy

import numpy as np
import pandas as pd
from sklearn import metrics


class PrivacyPreservingLogistic(object):
    def __init__(self, n, epsilon, l, dim, lamb, train_data, test_data,
                 init_network_state, step_size_scale, eval_period,
                 metric):
        self.n = n

        self.epsilon = epsilon

        self.l = l
        self.dim = dim
        self.lamb = lamb

        self.train_data = train_data        # {'X': design matrix, 'y': label vector}
        self.test_data = test_data          # {'X': design matrix, 'y': label vector}

        self.partition_size = self.train_data['X'].shape[0] // self.n

        self.init_network_state = copy.deepcopy(init_network_state)
        self.network_state = copy.deepcopy(init_network_state)

        self.step_size_scale = step_size_scale

        self.iteration = 0

        self.train_pred_y = None
        self.test_pred_y = None

        self.eval_period = eval_period

        self.metric = metric

        self.algorithm_name = None

        self.train_metric_array = []
        self.test_metric_array = []
        self.iteration_array = []           # iteration at which auc is recorded
        self._evaluate()

        self.x_transpose_y = [None] * self.n         # X^T * y for nodes

        self._init()
        self._parse_data()

    def _init(self):
        self._gen_A()
        self._parse_epsilon()

    def _parse_data(self):
        pass

    def run(self):
        while self.iteration < self.l:
            self._iterate()
            self.iteration += 1
            print("{}'s {}-eps iter: {}".format(self.algorithm_name, self.epsilon, self.iteration))

            if self.iteration % self.eval_period == 0:
                self._evaluate()

        # self._output_weight()
        self.save_metric()

    def _iterate(self):
        pass

    def _parse_epsilon(self):
        pass

    def _gen_A(self):
        """
        ring graph Laplacian
        :return:
        """
        self.A = 1 / 2 * np.identity(self.n)
        for i in range(self.A.shape[0]):
            self.A[i, (i + 1) % self.n] = 1 / 4
            self.A[i, (i - 1) % self.n] = 1 / 4
        self.A_tilde = np.kron(self.A, np.identity(self.dim))           # kron(A, I)

    @staticmethod
    def gen_init_state(node_num, dim):
        return np.random.random((node_num * dim, 1))

    @staticmethod
    def read_data(file_path):
        original_data = pd.read_csv(file_path, index_col=0).values
        output_data = {'y': original_data[:, 0:1], 'X': original_data[:, 1:]}
        return output_data

    def _gradient(self):
        # compute local gradients
        gradient_array = []
        for i in range(self.n):
            node_gradient = self._node_gradient(i)
            gradient_array.append(node_gradient)

        # concatenate
        gradient = gradient_array[0]
        for i in range(1, len(gradient_array)):
            gradient = np.concatenate((gradient, gradient_array[i]))
        return gradient

    def _node_gradient(self, i):
        feature_matrix, label_vector = self._retrieve_node_train_data(i)

        node_state = self.network_state[i * self.dim:(i + 1) * self.dim, :]
        pi = 1 / (1 + np.exp(-feature_matrix.dot(node_state)))

        # record X^T * y
        if self.x_transpose_y[i] is None:
            self.x_transpose_y[i] = feature_matrix.transpose().dot(label_vector)

        node_gradient = self.x_transpose_y[i] - (pi.transpose().dot(feature_matrix)).transpose() - self.lamb * node_state
        return node_gradient

    def _retrieve_node_train_data(self, i):
        feature_matrix = self.train_data['X'][i * self.partition_size:(i + 1) * self.partition_size, :]
        label_vector = self.train_data['y'][i * self.partition_size:(i + 1) * self.partition_size, :]
        return feature_matrix, label_vector

    def _step_size(self):
        return self.step_size_scale / (self.iteration + 1)

    def _average_state(self):
        average_state = self.network_state[:self.dim]
        for i in range(1, self.n):
            average_state += self.network_state[i * self.dim:(i + 1) * self.dim]
        average_state /= self.n
        return average_state

    def _evaluate(self):
        if self.metric == 'auc':
            self._predict()
            self._evaluate_auc()
        elif self.metric == 'acc':
            self._predict(use_threshold=True)
            self._evaluate_acc()
        elif self.metric == 'loss':
            self._evaluate_loss()
        else:
            raise ValueError("invalid type of metric: {}".format(self.metric))

        print("train {} = {}".format(self.metric, self.train_metric_array[-1]))
        print("test {} = {}".format(self.metric, self.test_metric_array[-1]))

        self._record_iteration()

    def _evaluate_loss(self):
        """

        :return:
        """
        weight = self._average_state()

        train_loss = self.compute_loss(self.train_data['X'], self.train_data['y'], weight)
        self.train_metric_array.append(train_loss)

        test_loss = self.compute_loss(self.test_data['X'], self.test_data['y'], weight)
        self.test_metric_array.append(test_loss)

    @staticmethod
    def compute_loss(X, y, w):
        logit = X.dot(w)

        loss = 0
        for i in range(X.shape[0]):
            loss += np.float(y[i, 0] * logit[i, 0] - np.log(1 + np.exp(logit[i, 0])))

        return -loss

    def _evaluate_acc(self):
        """
        Accuracy
        :return:
        """
        train_acc = metrics.accuracy_score(self.train_data['y'], self.train_pred_y)
        self.train_metric_array.append(train_acc)

        test_acc = metrics.accuracy_score(self.test_data['y'], self.test_pred_y)
        self.test_metric_array.append(test_acc)

    def _predict(self, use_threshold=False, threshold=0.5):
        weight = self._average_state()
        self.train_pred_y = self.train_data['X'].dot(weight)
        self.test_pred_y = self.test_data['X'].dot(weight)

        if use_threshold:
            for i in range(self.train_pred_y.shape[0]):
                if self.train_pred_y[i, 0] >= threshold:
                    self.train_pred_y[i, 0] = 1
                else:
                    self.train_pred_y[i, 0] = 0
            for i in range(self.test_pred_y.shape[0]):
                if self.test_pred_y[i, 0] >= threshold:
                    self.test_pred_y[i, 0] = 1
                else:
                    self.test_pred_y[i, 0] = 0

    def _evaluate_auc(self):
        # over training set
        fpr, tpr, thresholds = metrics.roc_curve(self.train_data['y'], self.train_pred_y)
        auc = metrics.auc(fpr, tpr)
        self.train_metric_array.append(auc)

        # over testing set
        fpr, tpr, thresholds = metrics.roc_curve(self.test_data['y'], self.test_pred_y)
        auc = metrics.auc(fpr, tpr)
        self.test_metric_array.append(auc)

    def _record_iteration(self):
        self.iteration_array.append(self.iteration)

    @staticmethod
    def gen_normal_random_variable(shape, mean, std_dev):
        rv = np.random.normal(mean, std_dev, shape)
        return rv

    @staticmethod
    def gen_laplace_random_variable(shape, mean, lamb):
        rv = np.random.laplace(mean, lamb, shape)
        return rv

    def _project(self):
        """
        Project the state onto the radius-one ball
        :return:
        """
        for i in range(self.n):
            node_state = self.network_state[i * self.dim:(i + 1) * self.dim]
            node_state_norm = np.linalg.norm(node_state)
            if node_state_norm > 1:
                node_state /= node_state_norm

    def _output_weight(self):
        print("terminal weight = {}".format(self._average_state()))

    def save_metric(self):
        alg_name = self.algorithm_name + str(np.log10(self.epsilon))
        np.save(r'../data/' + alg_name + '-train-{}.npy'.format(self.metric), self.train_metric_array)
        np.save(r'../data/' + alg_name + '-test-{}.npy'.format(self.metric), self.test_metric_array)
        np.save(r'../data/' + alg_name + '-iter.npy', self.iteration_array)


class PlainDistributedLogistic(PrivacyPreservingLogistic):
    def __init__(self, n, epsilon, l, dim, train_data, test_data, init_network_state, step_size_scale, eval_period):
        super(PlainDistributedLogistic, self).__init__(n, epsilon, l, dim, train_data, test_data,
                                                       init_network_state, step_size_scale, eval_period)

    def _iterate(self):
        step_size = self._step_size()
        gradient = self._gradient()
        self.network_state = self.A_tilde.dot(self.network_state) + step_size * gradient
