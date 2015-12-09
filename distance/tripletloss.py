import numpy as np

import chainer
from chainer import function
import chainer.functions as F
from chainer.utils import type_check
from chainer import cuda


class TripletLoss(function.Function):

    """"""

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(
            in_types[0].dtype == np.float32,
            in_types[1].dtype == np.float32,
            in_types[2].dtype == np.float32,
            in_types[0].shape == in_types[1].shape,
            in_types[0].shape == in_types[2].shape
        )

    def forward_cpu(self, inputs):
        a, p, n = inputs  # anchor, positive, negative
        # NOTE on using max(0, ...)
        # the loss is < 0 if (a-n) > (a-p)
        # that's what we want -- we don't want it increase the loss (by using
        # abs(), and we don't want it to decrease the sum by leaving it < 0)
        self.Li = np.maximum(0, (a-p)*(a-p) - (a-n)*(a-n) + self.margin)
        return np.array(np.sum(self.Li) / a.size, dtype=a[0].dtype),

    def forward_gpu(self, inputs):
        a, p, n = inputs  # anchor, positive, negative
        self.Li = cuda.cupy.maximum(0, (a-p)*(a-p) - (a-n)*(a-n) + self.margin)
        # if self.Li.sum() / a.size > 1:
        # if cuda.cupy.isnan(self.Li.sum()):
            # import pdb; pdb.set_trace()
        return self.Li.sum() / a.size,

    def backward(self, inputs, gy):
        a, p, n = inputs  # anchor, positive, negative
        coeff = gy[0] * gy[0].dtype.type(2. / self.Li.shape[0])
        gx0 = coeff * self.Li * (n - p)
        gx1 = coeff * self.Li * (p - a)
        gx2 = coeff * self.Li * (a - n)
        return gx0, gx1, gx2


def triplet_loss(x0, x1, x2):
    """Triplet loss function."""
    return TripletLoss()(x0, x1, x2)


class TripletAccuracy(function.Function):

    """"""

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(
            in_types[0].dtype == np.float32,
            in_types[1].dtype == np.float32,
            in_types[2].dtype == np.float32,
            in_types[0].shape == in_types[1].shape,
            in_types[0].shape == in_types[2].shape
        )

    def forward_cpu(self, inputs):
        a, p, n = inputs  # anchor, positive, negative
        N = a.shape[0]
        self.Li = (a-p)*(a-p) + self.margin - (a-n)*(a-n)
        return np.array(self.Li[self.Li < 0].size / a.size, dtype=a[0].dtype),

    def forward_gpu(self, inputs):
        a, p, n = inputs  # anchor, positive, negative
        N = a.shape[0]
        self.Li = (a-p)*(a-p) + self.margin - (a-n)*(a-n)
        return (self.Li < 0).sum() / a.size,


def triplet_accuracy(x0, x1, x2):
    """Triplet loss function."""
    return TripletAccuracy()(x0, x1, x2)
