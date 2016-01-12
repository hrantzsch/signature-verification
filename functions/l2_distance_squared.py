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
        n = diff.shape[0]
        y = np.zeros(n, dtype=x0.dtype)
        for i in range(n):
            y[i] = diff[i].sum()
        return y,

    def forward_gpu(self, inputs):
        x0, x1 = inputs
        l2distancesquared_kernel = cuda.cupy.ReductionKernel(
            'T x , T y', 'T z', '(x-y) * (x-y)', 'a + b', 'z = a', '0', 'l2distancesquared'
        )
        return l2distancesquared_kernel(x0, x1, axis=1),

    def backward(self, inputs, grad_outputs):
        x, y = inputs
        xp = cuda.get_array_module(x)
        gw, = grad_outputs
        gx = xp.zeros_like(x, dtype=x.dtype)
        for i in range(x.shape[0]):
            gx[i] = 2 * (x[i] - y[i]) * gw[i]
        return gx, -gx

def l2_distance_squared(x0, x1):

    """L2 distance (a.k.a. Euclidean distance) function squared."""

    return L2DistanceSquared()(x0, x1)
