import numpy

import chainer
from chainer import function
import chainer.functions as F
from chainer.utils import type_check


class TripletLoss(function.Function):

    """"""

    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
        self.mse1 = F.MeanSquaredError()
        self.mse2 = F.MeanSquaredError()

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[2].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape,
            in_types[0].shape == in_types[2].shape
        )

    def forward_cpu(self, inputs):
        a, p, n = inputs  # anchor, positive, negative
        diff_ap = self.mse1.forward_cpu(tuple((a, p)))
        diff_an = self.mse2.forward_cpu(tuple((a, n)))
        dtype = diff_ap[0].dtype
        distance = numpy.absolute(diff_ap[0] - diff_an[0] + self.margin * a.shape[0])
        return numpy.array(distance, dtype=dtype),

    def forward_gpu(self, inputs):
        a, p, n = inputs  # anchor, positive, negative
        diff_ap = self.mse1.forward_cpu(tuple((a, p)))
        diff_an = self.mse2.forward_cpu(tuple((a, n)))
        distance = numpy.absolute(diff_ap[0] - diff_an[0] + self.margin * a.shape[0])
        return numpy.array(distance),

    def backward(self, inputs, gy):
        # TODO
        coeff = gy[0] * gy[0].dtype.type(2. / self.diff.size)
        gx0 = coeff * self.diff
        return gx0, -gx0


def triplet_loss(x0, x1, x2):
    """Triplet loss function.

    #This function computes mean squared error between two variables. The mean
    #is taken over the minibatch. Note that the error is not scaled by 1/2.

    """
    return TripletLoss()(x0, x1, x2)
