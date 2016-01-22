import numpy as np

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Sqrt(function.Function):

    """Mean squared error (a.k.a. Euclidean loss) function."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(
            in_types[0].dtype == np.float32
        )

    def forward_cpu(self, inputs):
        x0, = inputs
        self.root = np.sqrt(x0)
        return self.root,

    def forward_gpu(self, inputs):
        x0, = inputs
        self.root = cuda.cupy.sqrt(x0)
        return self.root,

    def backward(self, inputs, gy):
        coeff = 1.0 / 2 * self.root
        return coeff * gy[0],


def sqrt(x0):
    """Mean squared error function.

    Computes MSE to a fixed vector of 0s and 1s.

    """
    return Sqrt()(x0)
