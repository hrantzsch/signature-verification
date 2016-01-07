import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class L2DistanceSquared(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        # TODO once forward_gpu is done I can use np here
        xp = cuda.get_array_module(inputs)
        x0, x1 = inputs
        diff = x0 - x1
        diff *= diff
        n = diff.shape[0]
        self.y = xp.zeros(n, dtype=x0.dtype)
        for i in range(n):
            self.y[i] = diff[i].sum()
        return self.y,

    def forward_gpu(self, inputs):
        self.forward_cpu(inputs)  # TODO

    def backward(self, inputs, gy):
        # TODO
        return 2 * self.y

                # coeff = gy[0] * gy[0].dtype.type(2. / self.y.size)
                # gx0 = coeff * self.y
                # return gx0, -gx0
                #
                # a, p, n = inputs  # anchor, positive, negative
                # N = a.shape[0]
                # coeff = gy[0] * gy[0].dtype.type(2. / self.Li.shape[0])
                # gx0 = coeff * self.Li * (n - p)
                # gx1 = coeff * self.Li * (p - a)
                # gx2 = coeff * self.Li * (a - n)
                # return gx0, gx1, gx2


def l2_distance_squared(x0, x1):

    """L2 distance (a.k.a. Euclidean distance) function squared."""

    return L2DistanceSquared()(x0, x1)
