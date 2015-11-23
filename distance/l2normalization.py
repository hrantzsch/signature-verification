import numpy as np

from chainer import function
from chainer.utils import type_check
from chainer import cuda

xp = cuda.cupy


class L2Normalization(function.Function):

    """"""

    def __init__(self, scale=1):
        self.scale = scale

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == np.float32,
            x_type.ndim >= 2,
        )

    def forward_cpu(self, inputs):
        x, = inputs
        N = (x.shape[0])
        self.norm = xp.zeros(N, dtype=np.int32)
        # TODO there's a better way
        for i in range(N):
            self.norm[i] = x[i].dot(x[i]) * self.scale
        return self.norm,

    def forward_gpu(self, inputs):
        return self.forward_cpu(inputs)

    def backward(self, inputs, gy):
        # gradient is simply 2x * self.scale
        err = 2 * inputs * self.scale
        return err * gy[0],


def l2_normalization(x0, scale=1):
    """"""
    return L2Normalization(scale)(x0)
