# Credit to gnilson
# https://github.com/gnilson/ProbitScale
#
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import norm
import matplotlib.ticker as ticker
from numpy import *


class ProbitScale (mscale.ScaleBase):

    name = 'probit'

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self)
        return

    def get_transform(self):
        return ProbitScale.ProbitTransform()

    def limit_range_for_scale(self, vmin, vmax, minpos):
        # return 0.001, 0.999
        # return 0.001, 0.6
        return max(vmin, 0.001), min(vmax, 0.999)

    def set_default_locators_and_formatters(self, axis):

        axis.set_major_locator(ticker.FixedLocator(
            array([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5,
                   0.6, 0.7, 0.8, 0.9, 0.99, 0.999])))
        axis.set_major_formatter(FormatStrFormatter('%g'))
        return

    class ProbitTransform(mtransforms.Transform):

        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self):
            mtransforms.Transform.__init__(self)
            return

        def transform(self, a):

            masked = ma.masked_where((a <= 0) | (a >= 1), a)
            if masked.mask.any():
                for i in arange(0, len(masked)):
                    masked[i] = norm.ppf(masked[i]) if (
                        (masked[i] < 1) and (masked[i] > 0)) else masked[i]
                return masked
            return norm.ppf(a)

        # for matplotlib 1.3.1
        def transform_non_affine(self, a):
            return self.transform(a)

        def inverted(self):
            return ProbitScale.CDFTransform()

    class CDFTransform(mtransforms.Transform):

        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self):
            mtransforms.Transform.__init__(self)
            return

        def transform(self, a):

            return norm.cdf(a)

        # for matplotlib 1.3.1
        def transform_non_affine(self, a):
            return self.transform(a)

        def inverted(self):
            return ProbitScale.ProbitTransform()
