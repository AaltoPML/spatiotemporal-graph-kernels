import gpflow
from gpflow import Parameter
import tensorflow as tf

import scipy.linalg
import numpy as np

from . import utils


def get_matern_kernel(L, nu, kappa):
    N = L.shape[0]
    alpha = nu
    Id = np.eye(N)
    A = ((2 * nu / kappa**2) * Id + L)
    A = scipy.linalg.fractional_matrix_power(A, alpha / 2)
    A = tf.cast(A, dtype=tf.float64)
    kern = tf.matmul(A, A, adjoint_a=True)
    kern = tf.linalg.pinv(kern)
    return kern


class LaplacianKernel(gpflow.kernels.base.Kernel):
    def __init__(self, sparse_adj_mat, variance=1.0, normalized_laplacian=True):
        super().__init__()
        self.variance = Parameter(variance, transform=gpflow.utilities.positive(), name="variance")
        self.sparse_adj_mat = sparse_adj_mat
        self.laplacian = utils.get_laplacian(sparse_adj_mat, normalized_laplacian)
        self.cov = tf.matmul(self.laplacian, self.laplacian, adjoint_a=True)
        self.cov = tf.linalg.pinv(self.cov)

    def K(self, X, Y=None, presliced=False):
        X = tf.reshape(tf.cast(X, tf.int32), [-1])
        X2 = tf.reshape(tf.cast(Y, tf.int32), [-1]) if Y is not None else X

        cov = self.variance * self.cov
        cov = tf.gather(tf.gather(cov, X, axis=0), X2, axis=1)
        return cov

    def K_diag(self, X, presliced=False):
        return tf.linalg.diag_part(self.K(X, presliced=presliced))


class DiffusionKernel(gpflow.kernels.base.Kernel):
    def __init__(self, sparse_adj_mat, variance=1.0, beta=0.1, normalized_laplacian=True):
        super().__init__()
        self.variance = Parameter(variance, transform=gpflow.utilities.positive(), name="variance")
        self.beta = beta

        self.sparse_adj_mat = sparse_adj_mat
        self.laplacian = utils.get_laplacian(sparse_adj_mat, normalized_laplacian)
        self.cov = tf.linalg.expm(-self.beta * self.laplacian)

    def K(self, X, Y=None, presliced=False):
        X = tf.reshape(tf.cast(X, tf.int32), [-1])
        X2 = tf.reshape(tf.cast(Y, tf.int32), [-1]) if Y is not None else X

        cov = self.variance * self.cov
        cov = tf.gather(tf.gather(cov, X, axis=0), X2, axis=1)

        return cov

    def K_diag(self, X, presliced=False):
        return tf.linalg.diag_part(self.K(X, presliced=presliced))


class RandomWalkKernel(gpflow.kernels.base.Kernel):
    def __init__(self, sparse_adj_mat, variance=1.0):
        super().__init__()
        self.variance = Parameter(variance, transform=gpflow.utilities.positive(), name="variance")
        sparse_adj_mat[np.diag_indices(sparse_adj_mat.shape[0])] = 1.0
        self.sparse_P = utils.sparse_mat_to_sparse_tensor(sparse_adj_mat)
        self.sparse_P = self.sparse_P / sparse_adj_mat.sum(axis=1)
        self.cov = tf.sparse.sparse_dense_matmul(self.sparse_P, tf.sparse.to_dense(self.sparse_P), adjoint_b=True)

    def K(self, X, Y=None, presliced=False):
        X = tf.reshape(tf.cast(X, tf.int32), [-1])
        X2 = tf.reshape(tf.cast(Y, tf.int32), [-1]) if Y is not None else X
        cov = self.variance * self.cov
        cov = tf.gather(tf.gather(cov, X, axis=0), X2, axis=1)
        return cov

    def K_diag(self, X, presliced=False):
        return tf.linalg.diag_part(self.K(X, presliced=presliced))


class MaternKernel(gpflow.kernels.base.Kernel):
    def __init__(self, sparse_adj_mat, nu, kappa, variance=1.0, normalized_laplacian=True):
        super().__init__()
        self.variance = Parameter(variance, transform=gpflow.utilities.positive(), name="variance")
        self.nu = nu
        self.kappa = kappa

        self.normalized_laplacian = normalized_laplacian
        self.laplacian = utils.get_laplacian(sparse_adj_mat=sparse_adj_mat, normalized_laplacian=normalized_laplacian)

        self.matern_kernel = get_matern_kernel(self.laplacian, self.nu, self.kappa)

    def K(self, X, Y=None, presliced=False):
        X = tf.reshape(tf.cast(X, tf.int32), [-1])
        X2 = tf.reshape(tf.cast(Y, tf.int32), [-1]) if Y is not None else X

        cov = self.variance * self.matern_kernel
        cov = tf.gather(tf.gather(cov, X, axis=0), X2, axis=1)
        return cov

    def K_diag(self, X, presliced=False):
        return tf.linalg.diag_part(self.K(X, presliced=presliced))


class WaveKernel(gpflow.kernels.base.Kernel):
    def __init__(self, sparse_adj_mat, variance=1.0, beta=0.1, c1=1, c2=1):
        super().__init__()
        self.variance = Parameter(variance, transform=gpflow.utilities.positive(),
                                  name="variance")
        self.c1 = Parameter(c1, transform=gpflow.utilities.positive(),
                            name="c1")
        self.c2 = Parameter(c2, transform=gpflow.utilities.positive(),
                            name="c2")

        self.beta = beta

        self.laplacian = utils.get_normalized_laplacian(sparse_adj_mat)
        self.sqrt_lapl = tf.constant(scipy.linalg.sqrtm(self.laplacian.numpy()), dtype=tf.float64)

        self.sqrt_inv_lapl = tf.constant(
            np.linalg.pinv(self.sqrt_lapl), dtype=tf.float64)
        self.sin = self.sqrt_inv_lapl @ scipy.linalg.sinm(self.sqrt_lapl * self.beta)
        self.cov = None

    def K(self, X, Y=None, presliced=False):
        X = tf.reshape(tf.cast(X, tf.int32), [-1])
        X2 = tf.reshape(tf.cast(Y, tf.int32), [-1]) if Y is not None else X
        self.sin = self.sqrt_inv_lapl @ scipy.linalg.sinm(self.sqrt_lapl * self.c1)
        self.cov = self.variance * self.sin
        self.cov = tf.gather(tf.gather(self.cov, X, axis=0), X2, axis=1)
        return self.cov

    def K_diag(self, X, presliced=False):
        return tf.linalg.diag_part(self.K(X, presliced=presliced))
