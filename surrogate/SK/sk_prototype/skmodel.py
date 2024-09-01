import numpy as np
from scipy.linalg import cholesky, solve_triangular, solve

_all__ = ["StochasticKriging"]


class StochasticKriging:
    """
        Stochastic Kriging model for simulation data.

        This class implements a Stochastic Kriging model.

        Attributes:
            length_scale (float): Length scale parameter for RBF kernel.
            sigma_f (float): Signal variance for RBF kernel.
            sigma_noise (float): Noise variance in outputs.
            X_train (ndarray): Normalized training inputs.
            Y_train (ndarray): Training outputs.
            B (ndarray): Matrix of basis functions (trend components).
            Vhat (ndarray): Estimated intrinsic variances at design points.
            beta_hat (ndarray): Coefficients for the linear trend component.
            L (ndarray): Lower triangular Cholesky factor of the covariance matrix.
            Z (ndarray): Solution of the system LZ = Y_tilde.
            Q (ndarray): Solution of the system L^TQ = Z.
        """
    def __init__(self, length_scale=1.0, sigma_f=1.0, sigma_noise=0.5):
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_noise = sigma_noise
        self.X_train = None
        self.Y_train = None
        self.B = None
        self.Vhat = None
        self.beta_hat = None
        self.L = None
        self.Z = None
        self.Q = None

    def normalize(self, X):
        """
        Normalize the input data.
        Args:
            X (ndarray): Input data to be normalized.
        Returns:
            ndarray: Normalized input data.
        """
        self.minX = np.min(X, axis=0)
        self.maxX = np.max(X, axis=0)
        return (X - self.minX) / (self.maxX - self.minX)

    def rbf_kernel(self, X1, X2):
        """
        Compute the Radial Basis Function (RBF) kernel matrix.
        Args:
            X1 (ndarray): First input matrix.
            X2 (ndarray): Second input matrix.
        Returns:
            ndarray: The RBF kernel matrix.
        """
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f ** 2 * np.exp(-0.5 / self.length_scale ** 2 * sqdist)

    def train(self, X_train, Y_train, Vhat):
        """
        Train the Stochastic Kriging model.
        Args:
            X_train (ndarray): Training input data.
            Y_train (ndarray): Training output data.
            Vhat (ndarray): Estimated intrinsic variances at design points.
        """
        if not (Y_train.shape[0] == X_train.shape[0] == Vhat.shape[0]):
            raise ValueError("X_train, Y_train, and Vhat must have the same number of rows.")

        self.X_train = self.normalize(X_train)
        self.Y_train = Y_train
        self.Vhat = Vhat
        self.B = np.ones((len(self.X_train), 1))  # Assuming a constant mean model

        # Compute beta_hat (coefficients for the trend component)
        self.beta_hat = solve(self.B.T @ self.B, self.B.T @ self.Y_train)
        Y_tilde = self.Y_train - self.B @ self.beta_hat

        # Compute the kernel matrix (spatial covariance matrix)
        K = self.rbf_kernel(self.X_train, self.X_train)
        Sigma = K + self.sigma_noise ** 2 * np.eye(len(self.X_train))

        # Combine the spatial covariance matrix with the intrinsic variances
        self.L = cholesky(Sigma + np.diag(self.Vhat))

        # Solve LZ = Y_tilde for Z
        self.Z = solve_triangular(self.L, Y_tilde, lower=True)

        # Solve L^TQ = Z for Q
        self.Q = solve_triangular(self.L.T, self.Z)

    def predict(self, X_s):
        X_s = self.normalize(X_s)
        K_s = self.rbf_kernel(self.X_train, X_s)

        # Compute mean
        mu_s = self.B @ self.beta_hat + K_s.T @ self.Q

        # Compute variance
        v = solve_triangular(self.L, K_s, lower=True)
        K_ss = self.rbf_kernel(X_s, X_s)
        cov_s = K_ss - v.T @ v

        stdv_s = np.sqrt(np.diag(cov_s))
        return mu_s, stdv_s
