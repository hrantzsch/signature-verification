import numpy as np

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Rint(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(
            in_types[0].dtype == np.float32
        )

    def forward_cpu(self, inputs):
        x0, = inputs
        self.r = np.rint(x0)
        return self.r.astype(np.int32),

    def forward_gpu(self, inputs):
        x0, = inputs
        self.r = cuda.cupy.rint(x0)
        return self.r.astype(cuda.cupy.int32),

    def backward_cpu(self, inputs, gy):
        return gy[0].astype(np.float32),

    def backward_gpu(self, inputs, gy):
        return gy[0].astype(cuda.cupy.float32),


def rint(x0):
    return Rint()(x0)
