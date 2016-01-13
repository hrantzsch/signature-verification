import numpy as np

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class L2DistanceSquared(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == np.float32,
            in_types[1].dtype == np.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        x0, x1 = inputs
        diff = x0 - x1
        diff *= diff
        return np.array([diff[i].sum() for i in range(len(x0))],
                        dtype=x0.dtype),

    def forward_gpu(self, inputs):
        x0, x1 = inputs
        l2distancesquared_kernel = cuda.cupy.ReductionKernel(
            'T x , T y', 'T z', '(x-y) * (x-y)', 'a + b', 'z = a', '0',
            'l2distancesquared'
        )
        return l2distancesquared_kernel(x0, x1, axis=1),

    def backward_cpu(self, inputs, gw):
        x0, x1 = inputs
        gx = np.array([2 * (x0[i] - x1[i]) * gw[0][i] for i in range(len(x0))],
                      dtype=x0.dtype)
        return gx, -gx

    def backward_gpu(self, inputs, gw):
        x0, x1 = inputs
        gw0 = gw[0].reshape(len(gw[0]), 1).repeat(x0.shape[1], axis=1)
        kernel = cuda.elementwise(
            'T x0, T x1, T gw0',
            'T gx',
            'gx = 2 * (x0 - x1) * gw0',
            'l2distancesquared_bwd')
        gx = kernel(x0, x1, gw0)
        return gx, -gx


def l2_distance_squared(x0, x1):

    """L2 distance (a.k.a. Euclidean distance) function squared."""

    return L2DistanceSquared()(x0, x1)
