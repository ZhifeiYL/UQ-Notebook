"""
A customized model based on the Gpy package to implement the stochastic Kriging model
"""

__all__ = ["StochasticKrigingModel"]

import GPy

class StochasticKrigingModel(GPy.core.GP):
    """
    Stochastic Kriging Gaussian Process model for regression with heteroscedastic noise

    This class extends the GPy.core.GP class to include stochastic kriging features.
    :param X: input observations
    :param Y: observed values
    :param kernel: a GPy kernel (correlation function[gammaP] in Matlab)
    :param Y_metadata: observed values variance ([Vhat] in Matlab) (heteroscedastic noise)
    """

    def __init__(self, X, Y, kernel=None, Y_metadata=None):
        # Ensure input dimensions match
        assert X.shape[0] == Y.shape[0] == Y_metadata.shape[0], "X, Y, and Y_metadata must have the same number of rows."

        if kernel is None:
            # default kernel to RBF
            kernel = GPy.kern.RBF(input_dim=X.shape[1])

        # Heteroscedastic noise model
        likelihood = GPy.likelihoods.HeteroscedasticGaussian(variance=Y_metadata)

        super(StochasticKrigingModel, self).__init__(X, Y, kernel, likelihood, name='Stochastic Kriging')


