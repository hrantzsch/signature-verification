import collections
import numpy as np
import six

from chainer import cuda


class GradientPrinting(object):

    """Optimizer hook function for printing the current gradient l2 norm.
       Should help me find out sensible values for gradient clipping."""
    name = 'GradientPrinting'

    def _sum_sqnorm(self, arr):
        sq_sum = collections.defaultdict(float)
        for x in arr:
            with cuda.get_device(x) as dev:
                x = x.ravel()
                s = x.dot(x)
                sq_sum[int(dev)] += s
        return sum([float(i) for i in six.itervalues(sq_sum)])

    def __call__(self, opt):
        print("### gradient printing ###")
        norm = np.sqrt(self._sum_sqnorm([p.grad for p in opt.target.params()]))
        rate = 1.0 / norm
        print("norm: {}".format(norm))
