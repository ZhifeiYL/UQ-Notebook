import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Gaussian Process with RBF Kernel
class GaussianProcess:
    def __init__(self):
        self.length_scale = 1.0
        self.sigma_f = 1.0
        self.sigma_noise = 0.5

    def rbf_kernel(self, X1, X2):
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.length_scale**2 * sqdist)

    def train(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.K = self.rbf_kernel(X_train, X_train) + self.sigma_noise**2 * np.eye(len(X_train))
        self.L = np.linalg.cholesky(self.K)

    def predict(self, X_s):
        K_s = self.rbf_kernel(self.X_train, X_s)
        Lk = np.linalg.solve(self.L, K_s)
        mu_s = np.dot(Lk.T, np.linalg.solve(self.L, self.Y_train)).flatten()

        # Compute the variance at our test points.
        K_ss = self.rbf_kernel(X_s, X_s)
        s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
        stdv = np.sqrt(s2)

        return mu_s, stdv