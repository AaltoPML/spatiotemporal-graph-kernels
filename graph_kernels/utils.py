import os
import json
import random
import pickle
import networkx as nx
import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

import sklearn.metrics
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import gpflow
from gpflow import Parameter

from . import data_utils


def sparse_mat_to_sparse_tensor(sparse_mat):
    """
    Converts a scipy csr_matrix to a tensorflow SparseTensor.
    """
    coo = sparse_mat.tocoo()
    indices = np.stack([coo.row, coo.col], axis=-1)
    tensor = tf.sparse.SparseTensor(indices, sparse_mat.data, sparse_mat.shape)
    return tensor


def normalize_laplacian(laplacian, d):
    inv_d = tf.linalg.diag([1. / float(el) if el != 0 else 0 for el in tf.linalg.diag_part(d)])
    inv_d = tf.cast(inv_d, dtype=tf.float64)
    inv_sqrt_d = tf.pow(inv_d, 0.5)
    laplacian_normalized = tf.linalg.matmul(inv_sqrt_d, laplacian)
    laplacian_normalized = tf.linalg.matmul(laplacian_normalized, inv_sqrt_d)
    return laplacian_normalized


def get_non_normalized_laplacian(sparse_adj_mat):
    sparse_adj_mat = sparse_mat_to_sparse_tensor(sparse_adj_mat)
    sparse_adj_mat = tf.cast(sparse_adj_mat, tf.float64)

    d_dense = tf.sparse.to_dense(tf.sparse.SparseTensor(
        indices=list(zip(*np.diag_indices(sparse_adj_mat.shape[0]))),
        values=tf.math.reduce_sum(tf.sparse.to_dense(sparse_adj_mat), axis=1),
        dense_shape=sparse_adj_mat.shape,
    ))
    laplacian_sparse = tf.math.subtract(
        d_dense, tf.sparse.to_dense(sparse_adj_mat))

    return laplacian_sparse, d_dense


def get_normalized_laplacian(sparse_adj_mat):
    laplacian_sparse, d_dense = get_non_normalized_laplacian(sparse_adj_mat)
    return normalize_laplacian(laplacian_sparse, d_dense)


def get_normalized_laplacian_from_graph(graph):
    return get_normalized_laplacian(
        nx.adjacency_matrix(graph, nodelist=range(len(graph.nodes())))
    )


def get_non_normalized_laplacian_from_graph(graph):
    return get_non_normalized_laplacian(
        nx.adjacency_matrix(graph, nodelist=range(len(graph.nodes())))
    )[0]


def get_laplacian(sparse_adj_mat, normalized_laplacian):
    if normalized_laplacian:
        return get_normalized_laplacian(sparse_adj_mat)
    else:
        return get_non_normalized_laplacian(sparse_adj_mat)[0]


def get_dataset_ids_from_graph(G, tr_ratio, random_seed=42):
    N = len(G.nodes())
    ids = np.array(list(range(N)))
    ids = shuffle(ids, random_state=random_seed)
    tr_id = int(N * tr_ratio)
    A = nx.to_scipy_sparse_matrix(G)
    idx_train, idx_test = ids[:tr_id, np.newaxis], ids[tr_id:, np.newaxis]
    return A, idx_train, idx_test


def evaluate_mse(X_val, y_val, gprocess):
    pred_y, pred_y_var = gprocess.predict_y(X_val)
    return sklearn.metrics.mean_squared_error(pred_y, y_val)


def evaluate_mape_predictions(pred_y, y_val, transformer=None):
    if transformer is not None:
        pred_y = transformer.inverse_transform(pred_y)
        y_val = transformer.inverse_transform(y_val)
    return sklearn.metrics.mean_absolute_percentage_error(y_val, pred_y)


def evaluate_mape(X_val, y_val, gprocess, transformer=None):
    pred_y, pred_y_var = gprocess.predict_y(X_val)
    return evaluate_mape_predictions(pred_y, y_val, transformer)


def evaluate_mae_predictions(pred_y, y_val, transformer=None):
    if transformer is not None:
        pred_y = transformer.inverse_transform(pred_y)
        y_val = transformer.inverse_transform(y_val)
    return sklearn.metrics.mean_absolute_error(y_val, pred_y)


def evaluate_mae(X_val, y_val, gprocess, transformer=None):
    pred_y, pred_y_var = gprocess.predict_y(X_val)
    return evaluate_mae_predictions(pred_y, y_val, transformer)


def smape(y_pred, y_true):
    return 100 / len(y_pred) * np.sum(2 * np.abs(y_true - y_pred) / (np.abs(y_pred) + np.abs(y_true)))


def plot(m, X_train, signal):
    xmin, xmax = 0.0, 30
    xx = np.linspace(xmin, xmax, 100)[:, None]
    mean, var = m.predict_y(xx)
    var = np.array([max(float(var[i]), 1e-3) for i in range(var.shape[0])])[:, np.newaxis]
    plt.figure(figsize=(12, 6))
    plt.plot(X_train, signal[[int(el) for el in X_train[:, 0]]], 'kx', mew=2)
    plt.plot(xx, mean, 'b', lw=2)
    plt.fill_between(xx[:, 0], mean[:, 0] - 2 * np.sqrt(var[:, 0]), mean[:, 0] + 2 * np.sqrt(var[:, 0]), color='blue', alpha=0.2)
    plt.xlim(xmin, xmax)
    plt.title("Adjacency matrix covariance function")


def visualize_gprocess(gprocess, X_train, X_test, G, signal, layout=None):
    X_all = tf.concat((X_train, X_test), axis=0)
    y_pred, var = gprocess.predict_y(X_all)

    y_pred_unshuffle = [0] * len(G.nodes())
    for i, y in zip(X_all, y_pred):
        y_pred_unshuffle[int(i)] = float(y)
    data_utils.plot_nodes_with_colors(G, y_pred_unshuffle, layout=layout)
    plot(gprocess, X_train, signal)


def training_step(X_train, y_train, optimizer, gprocess, natgrad=None):
    loss_fn = gprocess.training_loss_closure((X_train, y_train), compile=False)
    optimizer.minimize(loss_fn, var_list=gprocess.trainable_variables)
    if natgrad is not None:
        natgrad.minimize(loss_fn, var_list=[(gprocess.q_mu, gprocess.q_sqrt)])

    return -gprocess.elbo((X_train, y_train))


def cartesian_product(a, b):
    a_ = tf.reshape(tf.tile(a, [1, b.shape[0]]), (a.shape[0] * b.shape[0], a.shape[1]))
    b_ = tf.tile(b, [a.shape[0], 1])

    return tf.reshape(tf.concat([a_, b_], 1), [a.shape[0], b.shape[0], 4])


def is_pos_semi_def(x):
    return np.all(np.array(np.linalg.eigvals(x), dtype=np.float64) >= -1e-7)


def save_model_to_hyperparameters(model, save_path="gprocess_hyperparams.pkl"):
    pickle.dump(gpflow.utilities.parameter_dict(model), open(save_path, "wb"))


# loaded_result = loaded_model.predict_f_compiled(samples_input)
def load_model(model, path):
    params = pickle.load(open(path, "rb"))
    gpflow.utilities.multiple_assign(model, params)
    return model


def set_all_random_seeds(random_seed):
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()

    tf.random.set_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)


class ConstantArray(gpflow.mean_functions.MeanFunction):
    def __init__(self, shape):
        super().__init__()
        c = tf.zeros(shape)
        self.c = Parameter(c, name="constant array mean")

    def __call__(self, X):
        return tf.reshape(tf.gather(self.c, tf.cast(X[:, 0], dtype=tf.int32)), (X.shape[0], 1))


def tf_cosm(A):
    return tf.math.real(tf.linalg.expm(1j * tf.cast(A, dtype=tf.complex128)))


def tf_sinm(matrix):
    if matrix.dtype.is_complex:
        j_matrix = 1j * matrix
        return -0.5j * (tf.linalg.expm(j_matrix) - tf.linalg.expm(-j_matrix))
    else:
        j_matrix = tf.complex(tf.zeros_like(matrix), matrix)
        return tf.math.imag(tf.linalg.expm(j_matrix))


class Callback:
    def __init__(self, model, Xtrain, Ytrain, Xtest, Ytest, loss_fn=None, transformer=None):
        self.model = model
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest = Xtest
        self.Ytest = Ytest
        self.transformer = transformer
        self.epoch = 0
        self.loss_fn = loss_fn

    def __call__(self, step=None, variables=None, values=None):
        mape = evaluate_mape(self.Xtest, self.Ytest, self.model, transformer=self.transformer)
        mae = evaluate_mae(self.Xtest, self.Ytest, self.model, transformer=self.transformer)
        if self.loss_fn is None:
            elbo = self.model.elbo((self.Xtrain, self.Ytrain)).numpy()
        else:
            elbo = self.loss_fn()

        print(f"{self.epoch}:\tELBO: {elbo:.5f}\tMAPE: {mape:.10f}\tMAE: {mae:.10f}")
        self.epoch += 1


def replace_small_values(tensor, eps=1e-7):
    return tf.where(
        tf.abs(tensor) < eps,
        tf.ones_like(tensor), tensor)


def get_hmc_sample(num_samples, samples, hmc_helper, model, test_X):
    f_samples = []
    for i in range(num_samples):
        if i % 10 == 0:
            print(i)
        # Note that hmc_helper.current_state contains the unconstrained variables
        for var, var_samples in zip(hmc_helper.current_state, samples):
            var.assign(var_samples[i])
        f = model.predict_f_samples(test_X, 5)
        f_samples.append(f)
    f_samples = np.vstack(f_samples)
    return f_samples
