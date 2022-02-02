import os

import numpy as np
import pandas as pd
import networkx as nx
import pickle

import sklearn
from sklearn.preprocessing import FunctionTransformer

import matplotlib.pyplot as plt


HEAT_DATASET_1d = "./data/heat_distribution/1d.pkl"
HEAT_DATASET_2d = "./data/heat_distribution/2d.pkl"


def same_component(v1, v2, N):
    return v1 < N / 2 and v2 < N / 2 or v1 > N / 2 and v2 > N / 2


def get_noisy_signal(N, variance=0.1):
    signal = []
    for i in range(N // 2):
        signal.append(np.random.normal(-1, variance))

    signal.append(0)

    for i in range(N // 2):
        signal.append(np.random.normal(1, variance))
    return np.array(signal)


def generate_graph_n_comp(N, n_comp=3):
    raise NotImplementedError


def generate_graph(N, p):
    covariances = np.zeros((N, N))
    G = nx.Graph()
    for v1 in range(0, N):
        G.add_node(v1)
        for v2 in range(0, N):
            if v1 == v2:
                covariances[v1, v2] = 1
                continue
            else:
                if same_component(v1, v2, N) and np.random.rand() < p or v1 == N // 2 or v2 == N // 2:
                    G.add_edge(v1, v2)
                    covariances[v1, v2] = p
    return G, get_noisy_signal(N), covariances


def generate_ring_graph(N):
    graph = nx.Graph()
    graph.add_nodes_from(range(N))
    graph.add_edges_from([(i, (i + 1) % N) for i in range(N)])
    signal = [-1 + i * (2 / (N - 1)) for i in range(N)]
    return graph, np.array(signal)


def generate_lattice(n):
    G = nx.Graph()
    for v1 in range(0, n - 1):
        G.add_node(v1)
        G.add_node(v1 + 1)
        G.add_edge(v1, v1 + 1)
    return G


def generate_2d_lattice(n, m=None):
    if n is None:
        n = m
    return nx.generators.lattice.grid_2d_graph(m, n)


def draw_2d_lattice(G, signal=None):
    pos = dict((n, n) for n in G.nodes())
    labels = dict(((i, j), i * 10 + j) for i, j in G.nodes())
    nx.draw_networkx(G, pos=pos, labels=labels, node_color=signal)


def plot_nodes_with_colors(g, signal, title="2 component graph", layout=nx.spring_layout, ax=None):
    if layout is None:
        layout = nx.spring_layout

    nodes = g.nodes()
    assert len(nodes) == len(signal)

    # drawing nodes and edges separately so we can capture collection for colobar
    pos = layout(g)
    nx.draw_networkx_edges(g, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=signal,
                                node_size=100, cmap=plt.cm.jet, ax=ax)
    plt.title(title)
    plt.colorbar(nc)
    plt.axis('off')


def visualize_kernel_for_graph(gprocess, G, node=0, title=None):
    all_nodes = sorted(list(G.nodes()))
    covariance_train = gprocess.kernel.K(all_nodes)
    plot_nodes_with_colors(
        G,
        covariance_train[0],
        layout=nx.layout.circular_layout,
        title=title)


def read_heat_1d(path=HEAT_DATASET_1d):
    return pickle.load(open(path, "rb"))


def read_heat_2d(path=HEAT_DATASET_2d):
    return pickle.load(open(path, "rb"))


def build_graph_from_1d_points(x_lin):
    G = nx.Graph()
    for i in range(x_lin.shape[0]):
        G.add_node(i, point=x_lin[i])

    for i in range(1, x_lin.shape[0]):
        G.add_node(i, point=x_lin[i])
        G.add_edge(i - 1, i)
        if i + 1 < x_lin.shape[0]:
            G.add_edge(i, i + 1)

    return G


def build_graph_from_2d_points(X, Y):
    G = nx.Graph()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            G.add_node((i, j))
            if i - 1 >= 0:
                G.add_edge((i - 1, j), (i, j))
            if j - 1 >= 0:
                G.add_edge((i, j - 1), (i, j))
    return G


def generate_dataset(X, y, num_training_data, num_testing_data, start=0, log_target=False, rs=42,
                     interpolation=False):
    start_test = start + num_training_data
    end_test = start_test + num_testing_data

    if interpolation:
        train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(
            X[start:end_test], y[start:end_test],
            test_size=0.1, random_state=rs)
        train_y, test_y = train_y[:, np.newaxis], test_y[:, np.newaxis]
    else:
        train_X, train_y = X[start:start_test], y[start:start_test, np.newaxis]
        test_X, test_y = X[start_test:end_test], y[start_test:end_test, np.newaxis]

    if log_target:
        qt = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
        train_y = qt.fit_transform(train_y)
        test_y = qt.transform(test_y)
    else:
        qt = None

    return train_X, train_y, test_X, test_y, qt


def load_hungary_graph(path_to_csv="../data/hungary_chicken_pox/hungary_county_edges.csv"):
    df = pd.read_csv(path_to_csv)
    g = nx.from_pandas_edgelist(df, source="name_1", target="name_2")
    return g


def load_hungary_dataset(g, path_to_csv="../data/hungary_chicken_pox/hungary_chickenpox.csv"):
    df = pd.read_csv(path_to_csv)
    from_node_to_id = dict(zip(g.nodes(), range(len(g.nodes()))))
    X, y = [], []
    for t, row_dict in enumerate(df.to_dict(orient="records")):
        for v in g.nodes():
            X.append([from_node_to_id[v], t])
            y.append(row_dict[v])

    g = nx.relabel_nodes(g, from_node_to_id)
    return np.array(X), np.array(y), g, from_node_to_id


def generate_new_chickenpox_dataset(start=0):
    g = load_hungary_graph()
    X, y, g, node_ids = load_hungary_dataset(g)
    new_dataset = {}
    new_dataset["edges"] = list(g.edges())
    # new_dataset["node_ids"] = list(g.edges())
    new_dataset["FX"] = y[start * len(g.nodes()):].reshape((-1, len(g.nodes()))).tolist()
    return new_dataset


def generate_new_covid19_dataset(start=0):
    DATA_FOLDER = "../data/covid_data/"
    GRAPH_PATH = os.path.join(DATA_FOLDER, "g.pkl")

    g = pickle.load(open(GRAPH_PATH, "rb"))
    g = nx.relabel.convert_node_labels_to_integers(g)
    X_PATH = os.path.join(DATA_FOLDER, "X.pkl")
    Y_CASES_PATH = os.path.join(DATA_FOLDER, "y_cases.pkl")
    X = pickle.load(open(X_PATH, "rb"))
    y_cases = pickle.load(open(Y_CASES_PATH, "rb"))
    y_cases[y_cases < 0] = 0

    new_dataset = {}
    new_dataset["edges"] = list(g.edges())
    y_cases_reordered = []
    for i in range(start * len(g.nodes()), y_cases.shape[0], len(g.nodes())):
        cur = y_cases[i:i + len(g.nodes())]
        new_row = np.zeros(len(g.nodes()))
        for j, x in enumerate(X[i:i + len(g.nodes())]):
            new_row[int(x[0])] = cur[j]
        y_cases_reordered.append(new_row)
    y_cases_reordered = np.array(y_cases_reordered)
    new_dataset["FX"] = y_cases_reordered.tolist()
    return new_dataset
