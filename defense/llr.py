"""Compute log-likelihood ratio for a given distance.
   This module defines a class that encapsulates the fitted functions, so it
   can easily be loaded as a .pkl.
"""

import pickle
import numpy as np
from scipy import stats

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_palette('colorblind')
sns.set_color_codes("colorblind")


class LLR():

    def __init__(self, target_hist, nontarget_hist,
                 target_hist_edges, nontarget_hist_edges,
                 target_params, nontarget_params, max_dist):
        self.target_hist = target_hist
        self.nontarget_hist = nontarget_hist
        self.target_hist_edges = target_hist_edges
        self.nontarget_hist_edges = nontarget_hist_edges
        self.target_params = target_params
        self.nontarget_params = nontarget_params
        self.max_dist = max_dist

    def dist_to_score(self, dist):
        """Supposed to compute P(target_trial | s)"""
        return min(1, max(0, 2.5 * dist / self.max_dist))

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

    def show_histograms(self, highlight_dist=None):
        # fig, hist_plots = plt.subplots(2, figsize=(4, 3), dpi=360)
        fig, hist_plots = plt.subplots(2, figsize=(16, 12), dpi=360)

        colors = ['b' for _ in self.target_hist]
        if highlight_dist is not None:
            highlight_score = highlight_dist
            # highlight_score = self.dist_to_score(highlight_dist)
            highlight_idx = min(len(self.target_hist)-1,
                                np.digitize(highlight_score, self.target_hist_edges))

            # ensure that there no extra dims after pickling
            highlight_idx = np.squeeze(highlight_idx)

            colors[highlight_idx] = 'r'

        width = 1.0 * (self.target_hist_edges[1] - self.target_hist_edges[0])
        center = (self.target_hist_edges[:-1] + self.target_hist_edges[1:]) / 2

        hist_plots[0].set_ylim(0, 0.02)
        hist_plots[0].bar(center, self.target_hist,
                          align='center', width=width, color=colors)

        xx = np.linspace(0, self.max_dist, 500)
        yy = stats.exponweib.pdf(xx, *self.target_params)
        hist_plots[0].plot(xx, yy, 'g', lw=4)

        width = 1.0 * (self.nontarget_hist_edges[1] - self.nontarget_hist_edges[0])
        center = (self.nontarget_hist_edges[:-1] + self.nontarget_hist_edges[1:]) / 2

        hist_plots[1].set_ylim(0, 0.02)
        hist_plots[1].bar(center, self.nontarget_hist,
                          align='center', width=width, color=colors)

        yy = stats.exponweib.pdf(xx, *self.nontarget_params)
        hist_plots[1].plot(xx, yy, 'g', lw=4)

        hist_plots[0].set_title("Histogram target trials")
        hist_plots[1].set_title("Histogram non-target trials")

    def export(self, out):
        pickle.dump(self, open(out, 'wb'))

if __name__ == '__main__':
    import sys
    sys.path.append("..")
    llr_obj = pickle.load(open("llr_calibration.pkl", 'rb'))
    # print(llr_obj.llr(5.0))
    llr_obj.show_histograms(62.97)
    plt.show()
