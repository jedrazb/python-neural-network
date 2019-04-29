import numpy as np

from layer.layer import Layer


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        self._cache_current = None

    def _sigmoid(self, x):
        return 1/(1+(np.exp(-x)))

    def forward(self, x):

        self._cache_current = x
        return self._sigmoid(x)

    def backward(self, grad_z):

        x = self._cache_current
        f_x = self._sigmoid(x)
        return np.multiply(grad_z, f_x * (1 - f_x))
