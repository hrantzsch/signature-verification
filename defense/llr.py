"""Compute log-likelihood ratio for a given distance.
   This module defines a class that encapsulates the fitted functions, so it
   can easily be loaded as a .pkl.
"""

import pickle
import numpy as np
from scipy import stats


class LLR():

    def __init__(self, target_params, nontarget_params, max_dist):
        self.target_params = target_params
        self.nontarget_params = nontarget_params
        self.max_dist = max_dist

    def dist_to_score(self, dist):
        """Supposed to compute P(target_trial | s)"""
        return max(0, 2.5 * dist / self.max_dist)

    def llr(self, distance):
        """Compute the log-likelihood ratio."""
        s = self.dist_to_score(distance)
        l_target = stats.exponweib.pdf(s, *self.target_params)
        l_nontarget = stats.exponweib.pdf(s, *self.nontarget_params)
        if l_target == 0:
            return -np.inf
        if l_nontarget == 0:
            return np.inf
        return np.log(l_target / l_nontarget)

    def export(self, out):
        pickle.dump(self, open(out, 'wb'))

if __name__ == '__main__':
    import sys
    sys.path.append("..")
    llr_obj = pickle.load(open("llr_calibration.pkl", 'rb'))
    print(llr_obj.llr(5.0))
