import gpflow
from gpflow import Parameter
import tensorflow as tf

import scipy
from scipy import sparse

import networkx as nx

from . import utils
from . import kernels


def get_adj_matrix(graph):
    element = list(graph.nodes())[0]
    if nx.is_weighted(graph):
        return sparse.csr_matrix(nx.linalg.attrmatrix.attr_matrix(graph, "weight", rc_order=graph.nodes()))
    else:
        if isinstance(element, int):
            return nx.adjacency_matrix(graph, nodelist=range(len(graph.nodes())))
        else:
            return nx.adjacency_matrix(graph, nodelist=graph.nodes())


class TimeDistributed1dExponentialKernel(gpflow.kernels.base.Kernel):
    def __init__(self, graph, variance=1.0):
        super().__init__()
        self.variance = Parameter(variance, transform=gpflow.utilities.positive(), name="variance")
        self.graph = graph
        self.time_kernel = gpflow.kernels.Exponential()
        self.graph_kernel = gpflow.kernels.Exponential()
        gpflow.set_trainable(self.graph_kernel.variance, False)
        gpflow.set_trainable(self.time_kernel.variance, False)

    def K(self, X, Y=None, presliced=False):
        t = tf.reshape(tf.cast(X[:, -1], tf.float64), [X.shape[0], 1])
        X = tf.cast(X[:, :-1], tf.float64)

        if Y is not None:
            t2 = tf.reshape(tf.cast(Y[:, -1], tf.float64), [Y.shape[0], 1])
            X2 = tf.cast(Y[:, :-1], tf.float64)
        else:
            t2 = t
            X2 = X

        cov = self.variance * self.time_kernel(t, t2) * self.graph_kernel(X, X2)
        return cov

    def K_diag(self, X, presliced=False):
        return tf.linalg.diag_part(self.K(X, presliced=presliced))


class TimeDistributedGraphKernel(gpflow.kernels.base.Kernel):
    def __init__(self, graph, graph_kernel, variance=1.0, time_kernel_class=None):
        super().__init__()
        self.variance = Parameter(variance, transform=gpflow.utilities.positive(), name="variance")
        self.graph = graph
        self.time_kernel = time_kernel_class() if time_kernel_class is not None else gpflow.kernels.RBF()
        self.graph_kernel = graph_kernel
        gpflow.set_trainable(self.graph_kernel.variance, False)
        gpflow.set_trainable(self.time_kernel.variance, False)

    # Input: (node_id, time)
    def K(self, X, Y=None, presliced=False):
        t = tf.reshape(tf.cast(X[:, -1], tf.float64), [X.shape[0], 1])
        X = tf.cast(X[:, :-1], tf.float64)

        if Y is not None:
            t2 = tf.reshape(tf.cast(Y[:, -1], tf.float64), [Y.shape[0], 1])
            X2 = tf.cast(Y[:, :-1], tf.float64)
        else:
            t2 = t
            X2 = X

        cov = self.variance * self.time_kernel(t, t2) * self.graph_kernel(X, X2)
        return cov

    def K_diag(self, X, presliced=False):
        return tf.linalg.diag_part(self.K(X, presliced=presliced))


class TimeDistributedLaplacianKernel(TimeDistributedGraphKernel):
    def __init__(self, graph, variance=1.0, time_kernel_class=None, normalized_laplacian=True):
        sparse_adj_matrix = get_adj_matrix(graph)
        graph_kernel = kernels.LaplacianKernel(sparse_adj_matrix, normalized_laplacian=normalized_laplacian)
        super().__init__(graph, graph_kernel, variance, time_kernel_class)


class TimeDistributedMaternKernel(TimeDistributedGraphKernel):
    def __init__(self, graph, nu, kappa, variance=1.0, normalized_laplacian=True, time_kernel_class=None):
        sparse_adj_matrix = get_adj_matrix(graph)
        graph_kernel = kernels.MaternKernel(
            sparse_adj_matrix,
            nu, kappa, normalized_laplacian=normalized_laplacian)
        super().__init__(graph, graph_kernel, variance, time_kernel_class)


class TimeDistributedDiffusionKernel(TimeDistributedGraphKernel):
    def __init__(self, graph, variance=1.0, normalized_laplacian=True, time_kernel_class=None):
        sparse_adj_matrix = get_adj_matrix(graph)
        graph_kernel = kernels.DiffusionKernel(
            sparse_adj_matrix, normalized_laplacian=normalized_laplacian)
        super().__init__(graph, graph_kernel, variance, time_kernel_class)


class TimeDistributedRandomWalkKernel(TimeDistributedGraphKernel):
    def __init__(self, graph, variance=1.0, time_kernel_class=None):
        sparse_adj_matrix = get_adj_matrix(graph)
        graph_kernel = kernels.RandomWalkKernel(sparse_adj_matrix)
        super().__init__(graph, graph_kernel, variance, time_kernel_class)


def get_inds(t_indices, t2_indices, X, X2):
    # returns [(t1, t2, x1, x2)]
    left = tf.concat([t_indices, tf.cast(X, dtype=tf.int64)], axis=-1)
    right = tf.concat([t2_indices, tf.cast(X2, dtype=tf.int64)], axis=-1)
    inds = utils.cartesian_product(left, right)
    inds = tf.gather(inds, [0, 2, 1, 3], axis=2)
    return inds


def get_exponents_tf(vals, Gamma):
    return tf.linalg.expm(tf.tensordot(vals, -Gamma, axes=0))


def get_exponents(vals, Gamma):
    unique_vals = tf.sort(tf.unique(tf.reshape(vals, [-1]))[0])
    int_dist = tf.reshape(
        tf.where(
            tf.equal(tf.reshape(vals, [-1])[:, tf.newaxis], unique_vals[tf.newaxis, :]))[:, 1], vals.shape)
    unique_vals = tf.constant(unique_vals, dtype=tf.float64)
    return tf.gather(tf.linalg.expm(tf.tensordot(unique_vals, -Gamma, axes=0)), int_dist)


def get_exponents_scalar_tf(vals, lambdas):
    return tf.math.exp(tf.tensordot(-lambdas, vals, axes=0))


def get_exponents_scalar(vals, lambdas):
    unique_vals = tf.sort(tf.unique(tf.reshape(vals, [-1]))[0])
    int_dist = tf.reshape(
        tf.where(
            tf.equal(tf.reshape(vals, [-1])[:, tf.newaxis], unique_vals[tf.newaxis, :]))[:, 1], vals.shape)
    unique_vals = tf.constant(unique_vals, dtype=tf.float64)
    return tf.gather(tf.math.exp(tf.tensordot(-lambdas, unique_vals, axes=0)), int_dist, axis=1)


# calculating exp(-lambda (t + s))
def get_sums_exps(unique_t, unique_t2, lambdas):
    # we use only diagonal elements because consider diagonal matrix \Sigma
    unique_t, unique_t2 = tf.squeeze(unique_t), tf.squeeze(unique_t2)
    time_pairwise_sums = unique_t[:, None] + unique_t2[None, :]
    time_pairwise_sums = tf.tensordot(-lambdas, time_pairwise_sums, axes=0)
    return tf.math.exp(time_pairwise_sums)


# calculating a solution for stochastic heat equation
# for diagonal variance
def get_covariance_solution(dists_exps, sums_exps, variance, u, gamma_s):
    mult = tf.math.pow(tf.linalg.diag(variance), 2)
    pair_sums = utils.replace_small_values(gamma_s[None, :] + gamma_s[:, None], 1e-7)
    G = tf.linalg.diag_part(tf.math.divide(mult, pair_sums))[:, tf.newaxis, tf.newaxis, ] *\
        (dists_exps - sums_exps)
    G = tf.linalg.diag(tf.transpose(G, [1, 2, 0]))
    return u @ G @ tf.transpose(u)


def get_covariance_solution_fixed(t, s, u, variance, lambdas):
    sigma = tf.linalg.diag(variance)
    mult = tf.transpose(u) @ sigma @ tf.transpose(sigma) @ u
    pair_sums = lambdas[None, :] + lambdas[:, None]
    mult = tf.math.divide(mult, pair_sums)

    lt = lambdas[:, None] @ t[None, :]
    ls = lambdas[:, None] @ s[None, :]
    pairwise_sums = lt[:, :, None, None] + ls[None, None, :, :]
    pairwise_sums = tf.transpose(pairwise_sums, [0, 2, 1, 3])

    mins = tf.math.minimum(t[:, None], s[None, :])
    left = tf.math.exp(pair_sums[:, :, None, None] * mins[None, None, :, :] - pairwise_sums)

    right = tf.math.exp(-pairwise_sums)
    G = mult[:, :, None, None] * (left - right)
    G = tf.transpose(G, [2, 3, 0, 1])
    return u @ G @ tf.transpose(u)


class StochasticHeatEquation(gpflow.kernels.base.Kernel):
    def __init__(self, graph, variance=1.0, c=1, normalized_laplacian=True,
                 use_pseudodifferential=False, nu=None, kappa=None):
        super().__init__()
        self.variance = Parameter(variance, transform=gpflow.utilities.positive(1e-4), name="variance")
        self.c = Parameter(c, transform=gpflow.utilities.positive(1e-4), name="diffusion")
        self.graph = graph
        if nx.is_weighted(graph):
            self.laplacian = utils.get_laplacian(
                sparse.csr_matrix(nx.linalg.attrmatrix.attr_matrix(graph, "weight", rc_order=graph.nodes())),
                normalized_laplacian)
        else:
            self.laplacian = utils.get_laplacian(nx.adjacency_matrix(graph), normalized_laplacian)

        self.use_pseudodifferential = use_pseudodifferential
        if use_pseudodifferential:
            self.nu = nu
            self.kappa = kappa
        else:
            self.nu = None
            self.kappa = None

        # laplacian = self.u @ tf.linalg.diag(self.laplacian_s) @ tf.transpose(self.v)
        self.laplacian_s, self.u, self.v = tf.linalg.svd(self.laplacian)

    def get_scaled_differential_s(self):
        if self.use_pseudodifferential:
            return self.c * ((2 * self.nu) / (self.kappa ** 2) + self.laplacian_s) ** (self.nu / 2)
        else:
            return self.c * self.laplacian_s

    def get_scaled_differential(self):
        if self.use_pseudodifferential:
            return self.u @ tf.linalg.diag(self.get_scaled_differential_s()) @ tf.transpose(self.u)
        else:
            return self.c * self.laplacian

    # Input: (node_id, time)
    def K(self, X, Y=None, presliced=False):
        t = tf.reshape(tf.cast(X[:, -1], tf.float64), [X.shape[0]])
        X = tf.cast(X[:, :-1], tf.float64)
        if Y is not None:
            t2 = tf.reshape(tf.cast(Y[:, -1], tf.float64), [Y.shape[0]])
            X2 = tf.cast(Y[:, :-1], tf.float64)
        else:
            t2 = t
            X2 = X

        unique_t = tf.sort(tf.unique(t)[0])[:, tf.newaxis]
        unique_t2 = tf.sort(tf.unique(t2)[0])[:, tf.newaxis]

        self.time_pairwise_distances = tf.abs(unique_t - tf.transpose(unique_t2))
        self.time_pairwise_sums = (unique_t + tf.transpose(unique_t2))
        Gamma = self.get_scaled_differential()

        gamma_s = self.get_scaled_differential_s()
        if len(self.variance.shape) > 0:
            cov = get_covariance_solution_fixed(
                tf.squeeze(unique_t), tf.squeeze(unique_t2), self.u, self.variance, gamma_s)
        else:
            left_part = get_exponents(self.time_pairwise_distances, Gamma)
            right_part = get_exponents(self.time_pairwise_sums, Gamma)
            cov = self.variance * (left_part - right_part) @ tf.linalg.pinv(Gamma)
        t_indices = tf.where(tf.transpose(tf.equal(t, unique_t)))[:, 1]
        t2_indices = tf.where(tf.transpose(tf.equal(t2, unique_t2)))[:, 1]

        t_indices = tf.expand_dims(t_indices, 1)
        t2_indices = tf.expand_dims(t2_indices, 1)

        inds = get_inds(t_indices, t2_indices, X, X2)
        cov = tf.gather_nd(cov, inds)
        return cov

    def K_diag(self, X, presliced=False):
        return tf.linalg.diag_part(self.K(X, presliced=presliced))


def get_cosines(vals, Gamma):
    unique_vals = tf.sort(tf.unique(tf.reshape(vals, [-1]))[0])
    int_dist = tf.reshape(
        tf.where(
            tf.equal(tf.reshape(vals, [-1])[:, tf.newaxis], unique_vals[tf.newaxis, :]))[:, 1], vals.shape)
    result = utils.tf_cosm(tf.tensordot(unique_vals, Gamma, axes=0))
    return tf.gather(result, int_dist)


def get_sines(vals, Gamma):
    unique_vals = tf.sort(tf.unique(tf.reshape(vals, [-1]))[0])
    int_dist = tf.reshape(
        tf.where(
            tf.equal(tf.reshape(vals, [-1])[:, tf.newaxis], unique_vals[tf.newaxis, :]))[:, 1], vals.shape)
    result = utils.tf_sinm(tf.tensordot(unique_vals, Gamma, axes=0))
    return tf.gather(result, int_dist)


def get_cosines_tf(vals, Gamma):
    return utils.tf_cosm(tf.tensordot(vals, Gamma, axes=0))


def get_sines_tf(vals, Gamma):
    return utils.tf_sinm(tf.tensordot(vals, Gamma, axes=0))


class StochasticWaveEquationKernel(gpflow.kernels.base.Kernel):
    def __init__(self, graph, variance=1.0, c=1., normalized_laplacian=True, use_pseudodifferential=False,
                 nu=None, kappa=None):
        super().__init__()
        self.variance = Parameter(variance, transform=gpflow.utilities.positive(), name="variance")
        self.c = Parameter(c, transform=gpflow.utilities.positive(1e-2), name="propagation speed")
        self.graph = graph
        self.laplacian = utils.get_laplacian(nx.adjacency_matrix(graph), normalized_laplacian)
        if use_pseudodifferential:
            self.nu = nu
            self.kappa = kappa
            self.laplacian_s, self.u, self.v = tf.linalg.svd(self.laplacian)
        else:
            self.nu = None
            self.kappa = None
        self.id_l = tf.eye(self.laplacian.shape[0], dtype=tf.float64)

    # Input: (node_id, time)
    def K(self, X, Y=None, presliced=False):
        s = ((2 * self.nu) / (self.kappa ** 2) + self.laplacian_s) ** (self.nu / 2)
        self.laplacian = self.u @ tf.linalg.diag(s) @ tf.transpose(self.u)
        s = ((2 * self.nu) / (self.kappa ** 2) + self.laplacian_s) ** (self.nu / 4)
        self.sqrt_lapl = self.u @ tf.linalg.diag(s) @ tf.transpose(self.u)
        self.laplacian_inv = tf.linalg.pinv(self.laplacian)

        t = tf.reshape(tf.cast(X[:, -1], tf.float64), [X.shape[0]])
        X = tf.cast(X[:, :-1], tf.float64)
        if Y is not None:
            t2 = tf.reshape(tf.cast(Y[:, -1], tf.float64), [Y.shape[0]])
            X2 = tf.cast(Y[:, :-1], tf.float64)
        else:
            t2 = t
            X2 = X
        unique_t = tf.sort(tf.unique(t)[0])[:, tf.newaxis]
        unique_t2 = tf.sort(tf.unique(t2)[0])[:, tf.newaxis]
        time_pairwise_distances = tf.abs(unique_t - tf.transpose(unique_t2))

        theta = self.c * self.sqrt_lapl
        # Gamma = (self.c**2) * self.laplacian
        mins = tf.math.minimum(unique_t, tf.transpose(unique_t2))
        maxs = tf.math.maximum(unique_t, tf.transpose(unique_t2))
        # gamma_inv = tf.linalg.pinv(Gamma)
        gamma_inv = (1 / self.c**2) * self.laplacian_inv
        if len(self.variance.shape) > 0:
            raise Exception("Not implemented for matrix variance")
        else:
            gamma_inv = self.variance * gamma_inv
        cov = gamma_inv @ get_cosines(time_pairwise_distances, theta)
        cov = tf.tensordot(mins, self.id_l, axes=0) @ cov - 0.5 *\
            gamma_inv @ get_cosines(maxs, theta) @ get_sines(mins, theta) @ tf.linalg.inv(theta)

        t_indices = tf.where(tf.transpose(tf.equal(t, unique_t)))[:, 1]
        t2_indices = tf.where(tf.transpose(tf.equal(t2, unique_t2)))[:, 1]

        t_indices = tf.expand_dims(t_indices, 1)
        t2_indices = tf.expand_dims(t2_indices, 1)

        inds = get_inds(t_indices, t2_indices, X, X2)
        cov = tf.gather_nd(cov, inds)
        return cov

    def K_diag(self, X, presliced=False):
        return tf.linalg.diag_part(self.K(X, presliced=presliced))
