import os
import pickle
import json
import time
import argparse
import copy

from tqdm import tqdm

import gpflow

from graph_kernels import utils
from graph_kernels import time_kernels
from graph_kernels import data_utils
from graph_kernels import utils_opt


def parse_arguments():
    parser = argparse.ArgumentParser(description='COVID-19 across the US')
    parser.add_argument('--log_target', action='store_true', default=False,
                        help='Apply log transform to the target.')
    parser.add_argument('--no-log_target', dest='log_target', action='store_false')

    parser.add_argument('--use_flight_graph', action='store_true', default=False,
                        help='Use graph that contains information about the flights.')
    parser.add_argument('--no-use_flight_graph', dest='use_flight_graph', action='store_false')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--use_normalized_target', action='store_true', default=False,
                       help='Normalize the target by population in a state.')
    group.add_argument('--no-use_normalized_target', dest='use_normalized_target', action='store_false')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--interpolation', action='store_true', default=False,
                       help='Evaluate the models on the interpolation task.')
    group.add_argument('--extrapolation', dest='interpolation', action='store_false')

    parser.add_argument('--dump_directory', type=str, help='Path to directory with results.',
                        default="dump_directory")

    parser.add_argument('--num_test_weeks', type=int, help='Number of test weeks.', default=2)

    return parser.parse_args()


args = parse_arguments()

gpflow.config.set_default_jitter(1e-4)

DATA_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../data/covid_data/")

INTERPOLATION = args.interpolation
USE_FLIGHT_GRAPH = args.use_flight_graph
if USE_FLIGHT_GRAPH:
    GRAPH_PATH = os.path.join(DATA_FOLDER, "state_graph.pkl")
else:
    GRAPH_PATH = os.path.join(DATA_FOLDER, "g.pkl")
graph = pickle.load(open(GRAPH_PATH, "rb"))
N_NODES = len(graph.nodes())

NUM_TEST_WEEKS = args.num_test_weeks
NUM_TRAIN = 33 * N_NODES
NUM_TEST = NUM_TEST_WEEKS * N_NODES
START = 4 * N_NODES * 2
N_ITER = 5_000

IS_PREDICT_CASES = True
LOG_TARGET = args.log_target
USE_NORMALIZED_TARGET = args.use_normalized_target


RANDOM_SEEDS = [23, 42, 82, 100, 123,
                2 * 23, 2 * 42, 2 * 82, 2 * 100, 2 * 123]

X_PATH = os.path.join(DATA_FOLDER, "X.pkl")
if USE_NORMALIZED_TARGET:
    Y_CASES_PATH = os.path.join(DATA_FOLDER, "y_cases_normalized.pkl")
    Y_DEATHS_PATH = os.path.join(DATA_FOLDER, "y_deaths_normalized.pkl")
else:
    Y_CASES_PATH = os.path.join(DATA_FOLDER, "y_cases.pkl")
    Y_DEATHS_PATH = os.path.join(DATA_FOLDER, "y_deaths.pkl")


FROM_STATE_TO_ID_PATH = os.path.join(DATA_FOLDER, "from_state_to_id.pkl")
DUMP_DIRECTORY = args.dump_directory
DUMP_EVERYTHING = False
os.makedirs(DUMP_DIRECTORY, exist_ok=True)


X = pickle.load(open(X_PATH, "rb"))
y_cases = pickle.load(open(Y_CASES_PATH, "rb"))
y_deaths = pickle.load(open(Y_DEATHS_PATH, "rb"))
from_state_to_id = pickle.load(open(FROM_STATE_TO_ID_PATH, "rb"))


if IS_PREDICT_CASES:
    y = y_cases
else:
    y = y_deaths


y[y < 0] = 0

exp_kernels = {
    "td_matern_nu_52_k_1": time_kernels.TimeDistributedMaternKernel(graph, nu=5 / 2, kappa=1),
    "td_matern_nu_32_k_1": time_kernels.TimeDistributedMaternKernel(graph, nu=3 / 2, kappa=1),
    "td_matern_nu_12_k_1": time_kernels.TimeDistributedMaternKernel(graph, nu=1 / 2, kappa=1),
    "stoch_heat_vector_pseudo_diff_1": time_kernels.StochasticHeatEquation(
        graph, c=0.1, use_pseudodifferential=True, nu=5 / 2, kappa=1, variance=[1.] * len(graph.nodes())),
    "stoch_heat_vector_pseudo_diff_2": time_kernels.StochasticHeatEquation(
        graph, c=0.1, use_pseudodifferential=True, nu=3 / 2, kappa=1, variance=[1.] * len(graph.nodes())),
    "stoch_heat_vector_pseudo_diff_3": time_kernels.StochasticHeatEquation(
        graph, c=0.1, use_pseudodifferential=True, nu=1 / 2, kappa=1, variance=[1.] * len(graph.nodes())),
}


if __name__ == "__main__":
    results = {}
    for kernel_name, kernel in exp_kernels.items():
        results[kernel_name] = {}
        for i, rs in tqdm(enumerate(RANDOM_SEEDS), total=len(RANDOM_SEEDS)):
            utils.set_all_random_seeds(rs)
            train_X, train_y, test_X, test_y, qt = data_utils.generate_dataset(
                X, y, NUM_TRAIN, NUM_TEST,
                start=START + i * len(graph.nodes()), log_target=LOG_TARGET, rs=rs,
                interpolation=INTERPOLATION)
            print("Evaluating kernel ", kernel_name)
            start = time.time()
            result, gprocess = utils_opt.evaluate_kernel_mcmc(
                copy.deepcopy(kernel), train_X, train_y, test_X, test_y, graph,
                transformer=qt,
                n_iter=N_ITER, dump_directory=DUMP_DIRECTORY,
                dump_everything=DUMP_EVERYTHING, optimizer_name="LBFGS")
            results[kernel_name][rs] = result
            results[kernel_name][rs]["time"] = time.time() - start
        json.dump(results, open(os.path.join(DUMP_DIRECTORY, "results.json"), "w"))
