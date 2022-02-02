import os
import collections
import json
import numpy as np
import scipy
import scipy.stats


def load_result(path, filename="result.json"):
    path = os.path.join(path, filename)
    result = json.load(open(path, "rb"))
    return {int(k): v for k, v in result.items()}


def separate_results(result):
    all_iterations = []
    all_elbos = []
    all_mses = []
    for random_seed in result:
        cur_result = {int(k): v for k, v in result[random_seed].items()}
        iterations = sorted([int(i) for i in cur_result.keys()])

        all_iterations.extend(iterations)
        all_elbos.extend([cur_result[i]["ELBO"] for i in iterations])
        all_mses.extend([cur_result[i]["MSE"] for i in iterations])
    return all_iterations, all_elbos, all_mses


def separate_results_all_with_iter(result):
    all_metrics = collections.defaultdict(list)
    for random_seed in result:
        if "time" in result[random_seed]:
            del result[random_seed]["time"]
        cur_result = {int(k): v for k, v in result[random_seed].items()}
        iterations = sorted([int(i) for i in cur_result.keys()])

        all_metrics["iterations"].extend(iterations)
        for k in cur_result[0].keys():
            all_metrics[k].extend([cur_result[i][k] for i in iterations])
    return all_metrics


def separate_results_all(result):
    all_metrics = collections.defaultdict(list)
    for random_seed in result:
        for k in result[random_seed].keys():
            all_metrics[k].append(result[random_seed][k])
    return all_metrics


def stats_array(data):
    mean = np.mean(data)
    # evaluate sample variance by setting delta degrees of freedom (ddof) to
    # 1. The degree used in calculations is N - ddof
    stddev = np.std(data, ddof=1)
    # Get the endpoints of the range that contains 95% of the distribution
    t_bounds = scipy.stats.t.interval(0.95, len(data) - 1)
    # sum mean to the confidence interval
    ci = [mean + critval * stddev / (len(data)**0.5) for critval in t_bounds]
    print("Mean: {:.4f} $\\pm$ {:.4f}".format(mean, ci[1] - mean))
    print("Confidence Interval 95%%: {}, {}".format(ci[0], ci[1]))
    print(scipy.stats.t.interval(0.95, len(data) - 1, loc=np.mean(data), scale=scipy.stats.sem(data)))
    print("Data: ", data)


def print_statistics(result, n=2000):
    data = []
    for i, m in zip(result[0], result[2]):
        if i == n - 1:
            data.append(m)
    stats_array(data)


def from_folder_to_results(folder):
    results_dict = {}
    for dir_name in os.listdir(folder):
        results_dict[dir_name] = load_result(os.path.join(folder, dir_name))
    return results_dict


def parse_results(results):
    parsed_result = {}
    for k, v in results.items():
        parsed_result[k] = separate_results_all(v)
    return parsed_result
