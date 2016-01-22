import numpy

from chainer import function
from chainer.utils import type_check


class MseZeroOne(function.Function):

    """Mean squared error (a.k.a. Euclidean loss) function."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(
            in_types[0].dtype == numpy.float32
        )

    def forward_cpu(self, inputs):
        x0, = inputs
        x1 = numpy.array([0, 1] * len(x0), dtype=x0.dtype).reshape(len(x0), 2)
        self.diff = x0 - x1
        diff = self.diff.ravel()
        return numpy.array(diff.dot(diff) / diff.size, dtype=diff.dtype),

    def forward_gpu(self, inputs):
        x0, = inputs
        x1 = cuda.cupy.array([0, 1] * len(x0), dtype=x0.dtype).reshape(len(x0), 2)
        self.diff = x0 - x1
        diff = self.diff.ravel()
        return diff.dot(diff) / diff.dtype.type(diff.size),

    def backward(self, inputs, gy):
        coeff = gy[0] * gy[0].dtype.type(2. / self.diff.size)
        gx0 = coeff * self.diff
        return gx0,


def mse_zero_one(x0):
    """Mean squared error function.

    Computes MSE to a fixed vector of 0s and 1s.

    """
    return MseZeroOne()(x0)
