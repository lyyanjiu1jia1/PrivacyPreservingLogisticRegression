import numpy as np
from matplotlib import pyplot as plt
from execute.runner import metric

from algorithm import ALGORITHMS


partition_num = 40
dense_before = 300
x_lim = (0, 3000)
y_lim = (0.35, 0.9)

linewidth = 1
markersize = 3.5
linetype_train = {ALGORITHMS[0]: 'gs', ALGORITHMS[1]: 'rs', ALGORITHMS[2]: 'bs'}
linetype_test = {ALGORITHMS[0]: 'g^', ALGORITHMS[1]: 'r^', ALGORITHMS[2]: 'b^'}
# linetype_train = {ALGORITHMS[0]: 'g-', ALGORITHMS[1]: 'r-', ALGORITHMS[2]: 'b-'}
# linetype_test = {ALGORITHMS[0]: 'g.', ALGORITHMS[1]: 'r.', ALGORITHMS[2]: 'b.'}


def file_path_train(alg_name, epsilon):
    return r'../data/' + alg_name + str(np.log10(epsilon)) + '-train-' + metric + '.npy'


def file_path_test(alg_name, epsilon):
    return r'../data/' + alg_name + str(np.log10(epsilon)) + '-test-' + metric + '.npy'


def file_path_iter(alg_name, epsilon):
    return r'../data/' + alg_name + str(np.log10(epsilon)) + '-iter.npy'


def process_alg_name(alg_name):
    if alg_name == 'ppsc':
        return alg_name.upper() + '-Gossip'
    else:
        return alg_name.capitalize()


def even_spread(auc, iteration):
    auc_min = min(auc)
    auc_max = max(auc)
    auc_partition = np.linspace(auc_min, auc_max, partition_num).tolist()

    auc_output = []
    iter_output = []

    index = 0

    while auc_partition:
        lower = auc_partition.pop(0)
        while auc[index] < lower:
            index += 1

        if index == 0 or auc[index] != auc_output[-1]:
            auc_output.append(auc[index])
            iter_output.append(iteration[index])

    return auc_output, iter_output


def even_spread_huang(auc, iteration, dense_before=None):
    indexes = np.linspace(0, len(iteration), partition_num).tolist()

    if dense_before is not None:
        split = 0
        while indexes[split] <= dense_before:
            split += 1
        indexes = [i for i in range(split)] + indexes[split:]

    auc_output = []
    iter_output = []

    for index in indexes:
        if int(index) >= len(iteration):
            continue
        auc_output.append(auc[int(index)])
        iter_output.append(iteration[int(index)])

    return auc_output, iter_output


def plot_algorithm_subplot(subfig, alg_name, epsilon):
    """
    subfig corresponds to a specific epsilon
    :param subfig:
    :param alg_name:
    :param epsilon:
    :return:
    """
    # load
    train_metric = np.load(file_path_train(alg_name, epsilon))
    test_metric = np.load(file_path_test(alg_name, epsilon))
    iteration = np.load(file_path_iter(alg_name, epsilon))

    # process
    # if alg_name == ALGORITHMS[1]:
    #     train_metric, train_iter = even_spread_huang(train_metric, iteration)
    #     test_metric, test_iter = even_spread_huang(test_metric, iteration)
    # else:
    #     train_metric, train_iter = even_spread(train_metric, iteration)
    #     test_metric, test_iter = even_spread(test_metric, iteration)
    train_metric, train_iter = even_spread_huang(train_metric, iteration)
    test_metric, test_iter = even_spread_huang(test_metric, iteration)

    # plot
    subfig.plot(train_iter, train_metric, linetype_train[alg_name], linewidth=linewidth, markersize=markersize,
                label=process_alg_name(alg_name) + ' train')
    subfig.plot(test_iter, test_metric, linetype_test[alg_name], linewidth=linewidth, markersize=markersize,
                label=process_alg_name(alg_name) + ' test')

    # settings
    subfig.grid(True)
    subfig.set_title("$\epsilon={}$".format(epsilon))


def plot_algorithm(alg_name, epsilon, fig_index, early_dense=False, compensate_discontinuity=False):
    # load
    train_metric = np.load(file_path_train(alg_name, epsilon))
    test_metric = np.load(file_path_test(alg_name, epsilon))
    iteration = np.load(file_path_iter(alg_name, epsilon))

    # process
    # if alg_name == ALGORITHMS[1]:
    #     train_metric, train_iter = even_spread_huang(train_metric, iteration)
    #     test_metric, test_iter = even_spread_huang(test_metric, iteration)
    # else:
    #     train_metric, train_iter = even_spread(train_metric, iteration)
    #     test_metric, test_iter = even_spread(test_metric, iteration)
    if early_dense:
        train_metric, train_iter = even_spread_huang(train_metric, iteration, dense_before)
        test_metric, test_iter = even_spread_huang(test_metric, iteration, dense_before)
    else:
        train_metric, train_iter = even_spread_huang(train_metric, iteration)
        test_metric, test_iter = even_spread_huang(test_metric, iteration)

    if compensate_discontinuity:
        def compensate(array, compensate_num):
            first_elem = array.pop(0)
            second_elem = array.pop(0)
            array = np.linspace(first_elem, second_elem, compensate_num).tolist() + array
            return array

        compensate_num = 10
        train_metric = compensate(train_metric, compensate_num)
        train_iter = compensate(train_iter, compensate_num)
        test_metric = compensate(test_metric, compensate_num)
        test_iter = compensate(test_iter, compensate_num)

    # plot
    plt.figure(fig_index)
    plt.plot(train_iter, train_metric, linetype_train[alg_name], linewidth=linewidth, markersize=markersize,
             label=process_alg_name(alg_name) + ' training')
    plt.plot(test_iter, test_metric, linetype_test[alg_name], linewidth=linewidth, markersize=markersize,
             label=process_alg_name(alg_name) + ' test')

    # settings
    plt.grid(True)
    plt.title("Training / Test {} for $\epsilon={}$".format(process_metric_name(metric), epsilon))
    plt.xlabel('iteration $l$', fontsize=14)
    plt.ylabel(process_metric_name(metric), fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


def process_metric_name(metric_name):
    if metric_name == 'auc':
        return metric_name.upper()
    elif metric_name == 'acc':
        return 'Accuracy'
    elif metric_name == 'loss':
        return 'Loss'


def plot_all_in_subfigures():
    # setup
    epsilon_array = [1e-3, 1e-2, 1e-1]

    # create figures
    fig, subfigs = plt.subplots(1, 3)

    # plot
    for i in range(len(epsilon_array)):
        plot_algorithm_subplot(subfigs[i], ALGORITHMS[0], epsilon_array[i])

    for i in range(len(epsilon_array)):
        plot_algorithm_subplot(subfigs[i], ALGORITHMS[1], epsilon_array[i])

    for i in range(len(epsilon_array)):
        plot_algorithm_subplot(subfigs[i], ALGORITHMS[2], epsilon_array[i])

    # settings
    subfigs[1].set_xlabel('iteration $l$')
    subfigs[0].set_ylabel('Training / Test {}'.format(process_metric_name(metric)))

    subfigs[2].legend(loc='best')

    subfigs[0].set_xlim((0, 3000))
    subfigs[1].set_xlim((1, 3000))
    subfigs[2].set_xlim((2, 3000))

    plt.savefig(r'../figure/auc_subplots.png')


def plot_all():
    # setup
    epsilon_array = [1e-3, 1e-2, 1e-1]

    # plot
    for i in range(len(epsilon_array)):
        plot_algorithm(ALGORITHMS[0], epsilon_array[i], i)

    for i in range(len(epsilon_array)):
        plot_algorithm(ALGORITHMS[1], epsilon_array[i], i)

    for i in range(len(epsilon_array)):
        plot_algorithm(ALGORITHMS[2], epsilon_array[i], i, compensate_discontinuity=True)

    # settings
    for i in range(len(epsilon_array)):
        plt.figure(i)
        plt.legend(loc='best')
        plt.xlim(x_lim)
        plt.ylim(y_lim)

    # save
    for i in range(len(epsilon_array)):
        plt.figure(i)
        plt.savefig(r'../figure/{}'.format(metric) + str(np.log10(epsilon_array[i])).replace('.', '-') + '.png')


# plot_all_in_subfigures()
plot_all()
