import numpy as np

from layer.layer import Layer


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        self._cache_current = None

    def forward(self, x):

        self._cache_current = x
        return x * (x > 0)

    def backward(self, grad_z):

        return grad_z * (self._cache_current > 0)
