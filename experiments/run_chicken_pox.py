import os

import json
import time
import argparse
import copy
import sklearn

import networkx as nx
from tqdm import tqdm

import gpflow
import tensorflow as tf

from graph_kernels import utils
from graph_kernels import data_utils
from graph_kernels import utils_opt
from graph_kernels import time_kernels

gpflow.config.set_default_jitter(1e-4)
gpflow.config.set_default_float(tf.float64)
f64 = gpflow.utilities.to_default_float


def parse_arguments():
    parser = argparse.ArgumentParser(description='Toy epidemiological dataset.')
    parser.add_argument('--dump_directory', type=str,
                        help='Path to directory with results.',
                        default="dump_directory")

    parser.add_argument('--num_test_weeks', type=int, help='Number of test weeks.', default=2)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--interpolation', action='store_true', default=False,
                       help='Evaluate the models on the interpolation task.')
    group.add_argument('--extrapolation', dest='interpolation', action='store_false')

    return parser.parse_args()


args = parse_arguments()

INTERPOLATION = args.interpolation
DATA_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../data/hungary_chicken_pox/")

GRAPH_PATH = os.path.join(DATA_FOLDER, "hungary_county_edges.csv")
graph = data_utils.load_hungary_graph(GRAPH_PATH)
graph.remove_edges_from(nx.selfloop_edges(graph))

NUM_TEST_WEEKS = args.num_test_weeks
NUM_TRAIN = 103 * len(graph.nodes())
NUM_TEST = len(graph.nodes()) * NUM_TEST_WEEKS
N_ITER = 2_000

RANDOM_SEEDS = [23, 42, 82, 100, 123, 223,
                2 * 23, 2 * 42, 2 * 82, 2 * 100, 2 * 123, 2 * 223]

DUMP_DIRECTORY = args.dump_directory
DUMP_EVERYTHING = False
os.makedirs(DUMP_DIRECTORY, exist_ok=True)


X, y, graph, _ = data_utils.load_hungary_dataset(
    graph, path_to_csv=os.path.join(DATA_FOLDER, "hungary_chickenpox.csv"))


exp_kernels = {
    #"td_laplacian": time_kernels.TimeDistributedLaplacianKernel(graph),
    "stoch_heat_vector_pseudo_diff_1_scalar": time_kernels.StochasticHeatEquation(
        graph, c=0.1, use_pseudodifferential=True, nu=1 / 2,
        kappa=1, variance=1.),
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


if __name__ == "__main__":
    results = {}
    for kernel_name, kernel in exp_kernels.items():
        print("Evaluating kernel ", kernel_name)
        results[kernel_name] = {}
        for i, rs in tqdm(enumerate(RANDOM_SEEDS), total=len(RANDOM_SEEDS)):
            utils.set_all_random_seeds(rs)
            train_X, train_y, test_X, test_y, qt = data_utils.generate_dataset(
                X, y.ravel(),
                num_training_data=NUM_TRAIN, num_testing_data=NUM_TEST,
                start=i * len(graph.nodes()),
                log_target=True, rs=rs)
            if INTERPOLATION:
                train_X, test_X, train_y, test_y = \
                    sklearn.model_selection.train_test_split(
                        train_X, train_y, test_size=0.1, random_state=rs)
            train_y = tf.cast(train_y, tf.float64)
            test_y = tf.cast(test_y, tf.float64)

            start = time.time()
            result, gprocess = utils_opt.evaluate_kernel_mcmc(
                copy.deepcopy(kernel), train_X, train_y, test_X, test_y, graph,
                transformer=qt,
                n_iter=N_ITER, optimizer_name="LBFGS")
            results[kernel_name][rs] = result
            results[kernel_name][rs]["time"] = time.time() - start
        json.dump(results, open(os.path.join(DUMP_DIRECTORY, "results.json"), "w"))
