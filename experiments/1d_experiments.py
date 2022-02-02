import argparse
import os
import json
import copy

import numpy as np
from tqdm import tqdm

import sklearn
import sklearn.metrics

import gpflow
from gpflow import Parameter
import tensorflow as tf

from graph_kernels import data_utils
from graph_kernels import time_kernels
from graph_kernels import utils_opt
from graph_kernels import utils


def parse_arguments():
    parser = argparse.ArgumentParser(description='Heat distribution over a 1d line.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--interpolation', action='store_true', default=False,
                       help='Evaluate the models on the interpolation task.')
    group.add_argument('--extrapolation', dest='interpolation', action='store_false')

    parser.add_argument('--dump_directory', type=str, help='Path to directory with results.',
                        default="dump_directory")
    return parser.parse_args()


args = parse_arguments()
INTERPOLATION = args.interpolation
DUMP_DIRECTORY = args.dump_directory
if not os.path.exists(DUMP_DIRECTORY):
    os.makedirs(DUMP_DIRECTORY)


DATASET_PATH_1d = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../data/heat_distribution/1d.pkl")
DATASET_PATH_2s = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../data/heat_distribution/2d.pkl")

N_ITER = 2000
RANDOM_SEEDS = [23, 42, 82, 100, 2 * 23, 2 * 42, 2 * 82, 2 * 100]
NUM_TRAIN = 50  # number of training timestamps
NUM_TEST = 10

gpflow.config.set_default_jitter(1e-8)


class ConstantArray(gpflow.mean_functions.MeanFunction):
    def __init__(self, shape):
        super().__init__()
        c = tf.zeros(shape)
        self.c = Parameter(c)

    def __call__(self, X):
        return tf.reshape(
            tf.gather(self.c, tf.cast(X[:, 0], dtype=tf.int32)),
            (X.shape[0], 1)
        )


def convert_dataset_to_nodes(data):
    new_data = []
    for row in data:
        new_data.append([from_x_to_nodes[row[0]], row[1]])
    return np.array(new_data)


def extract_ml_dataset(dataset_1d, times):
    data = []
    target = []
    for t in times:
        for i, el in enumerate(dataset_1d[t]["x"]):
            data.append(np.append(el, np.array(t)))
            target.append(dataset_1d[t]["y"][i])
    data = np.array(data)
    target = np.array(target)[:, np.newaxis]
    return data, target


def evaluate_kernel(kernel, kernel_name):
    results = {}
    for random_seed in tqdm(RANDOM_SEEDS):
        utils.set_all_random_seeds(random_seed)

        train_data, train_target, test_data, test_target = datasets[random_seed]
        if kernel_name != "td_exponential":
            train_data = convert_dataset_to_nodes(train_data)
            test_data = convert_dataset_to_nodes(test_data)
            mean_function = ConstantArray(num_nodes)
        else:
            mean_function = gpflow.mean_functions.Constant()
        print("Shape: ", train_data.shape, test_data.shape)
        result, gprocess = utils_opt.evaluate_kernel_mcmc(
            copy.deepcopy(kernel), train_data, train_target,
            test_data, test_target, graph, mean_function=copy.deepcopy(mean_function), n_iter=N_ITER,
            optimizer_name="LBFGS")

        results[random_seed] = result
    return results, gprocess


dataset_1d = data_utils.read_heat_1d(DATASET_PATH_1d)

t = list(dataset_1d.keys())[0]
graph = data_utils.build_graph_from_1d_points(dataset_1d[t]["x"])

num_nodes = len(graph.nodes())

from_x_to_nodes = {data["point"]: node for node, data in graph.nodes(data=True)}

times = sorted(dataset_1d.keys())[1:]

datasets = {}
for i, rs in enumerate(RANDOM_SEEDS):
    start = i
    start_testing = start + NUM_TRAIN
    train_times = times[start:start_testing]
    test_times = times[start_testing:start_testing + NUM_TEST]

    train_data, train_target = extract_ml_dataset(dataset_1d, train_times)
    test_data, test_target = extract_ml_dataset(dataset_1d, test_times)
    if INTERPOLATION:
        train_data, test_data, train_target, test_target = \
            sklearn.model_selection.train_test_split(
                np.concatenate((train_data, test_data)),
                np.concatenate((train_target, test_target)), test_size=0.1, random_state=rs)

    datasets[rs] = (train_data, train_target, test_data, test_target)


kernels = {
    "td_exponential": time_kernels.TimeDistributed1dExponentialKernel(graph),
    # "td_laplacian": time_kernels.TimeDistributedLaplacianKernel(graph),
    "td_matern_nu_52_d_1": time_kernels.TimeDistributedMaternKernel(graph, nu=5 / 2, kappa=1),
    "td_matern_nu_32_d_1": time_kernels.TimeDistributedMaternKernel(graph, nu=3 / 2, kappa=1),
    "td_matern_nu_12_d_1": time_kernels.TimeDistributedMaternKernel(graph, nu=1 / 2, kappa=1),
    "stoch_heat_vector_pseudo_diff_1": time_kernels.StochasticHeatEquation(
        graph, c=0.1, use_pseudodifferential=True, nu=5 / 2,
        kappa=1, variance=[1.] * len(graph.nodes())),
    "stoch_heat_vector_pseudo_diff_2": time_kernels.StochasticHeatEquation(
        graph, c=0.1, use_pseudodifferential=True, nu=3 / 2,
        kappa=1, variance=[1.] * len(graph.nodes())),
    "stoch_heat_vector_pseudo_diff_3": time_kernels.StochasticHeatEquation(
        graph, c=0.1, use_pseudodifferential=True, nu=1 / 2,
        kappa=1, variance=[1.] * len(graph.nodes())),
}

for kernel_name, kernel in kernels.items():
    print("Evaluating {}".format(kernel_name))
    result, gprocess = evaluate_kernel(
        copy.deepcopy(kernel), kernel_name)
    folder = os.path.join(DUMP_DIRECTORY, kernel_name)
    os.makedirs(folder, exist_ok=True)
    # pickle.dump(gprocess, open(os.path.join(folder, "gprocess.pkl"), "wb"))
    json.dump(result, open(os.path.join(folder, "result.json"), "w"))
