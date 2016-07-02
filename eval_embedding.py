"""A script to evaluate a model's embeddings.

The script expects embedded data as a .pkl file.

Currently the script prints min, mean, and max distances intra-class and
comparing a class's samples to the respective forgeries.
"""

import pickle
import sys
import numpy as np
from scipy.spatial.distance import cdist, pdist
from chainer import cuda


AVG = True
DIST_METHOD = 'sqeuclidean'


# ============================================================================

# [*1] HACK:
# scaling the genuine-forgery result by (num_genuines / num_forgeries) per user
# in order to balance the number for genuines and forgeries I have
# (see https://en.wikipedia.org/wiki/Accuracy_paradox)
# however num_genuines depends on whether or not I average the genuines:
# 5 if I do, 4 if I don't (not comparing a genuine with itself)
# gen_forge_dists[user].shape[0] is however always 5; thus, reduce by one if I
# don't take the average

# ============================================================================


def false_positives(gen_forge_dists, threshold):
    """Number of Genuine-Forgery pairs that are closer than threshold"""
    result = 0
    for user in gen_forge_dists.keys():
        user_result = np.sum(gen_forge_dists[user] <= threshold)
        user_result *= (gen_forge_dists[user].shape[0] - (1 - AVG)) / \
            gen_forge_dists[user].shape[1]  # HACK [*1]
        result += user_result
    return result


def true_negatives(gen_forge_dists, threshold):
    """Number of Genuine-Forgery pairs that are farther than threshold"""
    result = 0
    for user in gen_forge_dists.keys():
        user_result = np.sum(gen_forge_dists[user] > threshold)
        user_result *= (gen_forge_dists[user].shape[0] - (1 - AVG)) / \
            gen_forge_dists[user].shape[1]  # HACK [*1]
        result += user_result
    return result


def true_positives(gen_gen_dists, threshold):
    """Number of Genuine-Genuine pairs that are closer than threshold"""
    result = 0
    for user in gen_gen_dists.keys():
        dists = gen_gen_dists[user][gen_gen_dists[user] != 0]
        result += np.sum(dists <= threshold)
    return result


def false_negatives(gen_gen_dists, threshold):
    """Number of Genuine-Genuine pairs that are farther than threshold"""
    result = 0
    for user in gen_gen_dists.keys():
        dists = gen_gen_dists[user][gen_gen_dists[user] != 0]
        result += np.sum(dists > threshold)
    return result


# ============================================================================


def genuine_genuine_dists(data, average):
    gen_keys = [k for k in data.keys() if 'f' not in k]
    dists = {}
    for k in gen_keys:
        gen = cuda.cupy.asnumpy(data[k])
        if average:
            gen_mean = []
            for i in range(len(gen)):
                others = list(range(len(gen)))
                others.remove(i)
                gen_mean.append(np.mean(gen[others], axis=0))
            dists[k] = cdist(gen_mean, gen, DIST_METHOD)
        else:
            dists[k] = cdist(gen, gen, DIST_METHOD)
    return dists


def genuine_forgeries_dists(data, average):
    gen_keys = [k for k in data.keys() if 'f' not in k]
    dists = {}
    for k in gen_keys:
        gen = cuda.cupy.asnumpy(data[k])
        if average:
            gen_mean = []
            for i in range(len(gen)):
                others = list(range(len(gen)))
                others.remove(i)
                gen_mean.append(np.mean(gen[others], axis=0))
            gen = gen_mean
        forge = cuda.cupy.asnumpy(data[k + '_f'])
        # np.random.shuffle(forge)
        # forge = forge[:5]  # HACK reduce number of forgeries
        dists[k] = cdist(gen, forge, DIST_METHOD)
    return dists


# ============================================================================


def roc(thresholds, data):
    """Compute point in a ROC curve for given thresholds.
       Returns a point (false-pos-rate, true-pos-rate)"""
    gen_gen_dists = genuine_genuine_dists(data, AVG)
    gen_forge_dists = genuine_forgeries_dists(data, AVG)
    for i in thresholds:
        tp = true_positives(gen_gen_dists, i)
        fp = false_positives(gen_forge_dists, i)
        tn = true_negatives(gen_forge_dists, i)
        fn = false_negatives(gen_gen_dists, i)

        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        fnr = fn / (tp + fn)

        aer = (fpr + fnr) / 2  # Avg Error Rate

        if tp + fp == 0:
            # make sure we don't divide by zero
            f1 = 0.0
        else:
            precision = tp / (tp + fp)
            recall = tpr
            f1 = 2 * precision * recall / (precision + recall)

        acc = (tp + tn) / (tp + fp + fn + tn)

        yield (fpr, tpr, f1, acc, aer)


# ============================================================================


def print_stats(data):
    intra_class_dists = {k: pdists_for_key(data, k) for k in data.keys()}
    gen_forge_dists = genuine_forgeries_dists(data, False)

    print("class\t|\tintra-class\t\t|\tforgeries")
    print("-----\t|\tmin - mean - max\t|\tmin - mean - max")
    for k in sorted(intra_class_dists.keys()):
        if 'f' in k:
            continue
        dist = intra_class_dists[k]
        print("{}\t|\t{} - {} - {}".format(k,
                                           int(np.min(dist)),
                                           int(np.mean(dist)),
                                           int(np.max(dist))),
              end='\t\t')
        dist = gen_forge_dists[k]
        print("|\t{} - {} - {}".format(int(np.min(dist)),
                                       int(np.mean(dist)),
                                       int(np.max(dist))))


# ============================================================================


def load_keys(dictionary, keys):
    values = [s for k in keys for s in dictionary[k]]
    return np.stack(list(map(cuda.cupy.asnumpy, values)))


def pdists_for_key(embeddings, k):
    samples = cuda.cupy.asnumpy(embeddings[k])
    return pdist(samples, DIST_METHOD)


# ============================================================================


def dist_to_score(dist, max_dist):
    """Supposed to compute P(target_trial | s)"""
    return max(1 / max_dist, 1 - dist / max_dist)


def likelihood(score, bins, bin_edges):
    """Compute the likelihood to observe a score in a (non)target trial, i.e.
       P(s | (non)target_trial).
       `bins` and `bin_edges` should be as obtained from numpy.histogram of the
       (non)target trial data."""
    # match sample to bin from histogram data
    score_bin = np.digitize(score, bin_edges)
    # likelihood is size_bin / size_all_bins
    try:
        return bins[score_bin] / np.sum(bins)
    except IndexError:  # happens for scores smaller/larger than any we've seen
        return 0.0


def llr(score,
        target_bins, target_bin_edges,
        nontarget_bins, nontarget_bin_edges):
    return np.log(likelihood(score, target_bins, target_bin_edges /
                  likelihood(score, nontarget_bins, nontarget_bin_edges)))


# def logit(x):
#     return np.log(x / (1 - x))


# def llr(dist, condition_positive_rate):
#     """Computes log-Likelihood-ratio given a sample's distance and the ratio of
#        condition positive samples in the data."""
#     # TODO provide a max_dist or find a more intelligent way
#     return logit(dist_to_score(dist)) - logit(condition_positive_rate)


# def dist_to_conf(dist, thresh, mode='step'):
#     # TODO: I want llr, not conf. How to get the llr from the probability?
#     # log-likelihood-ratio: This is the logarithm of the ratio between the
#     # likelihood that the target produced the speech input, and the likelihood
#     # that a non-target produced the input.
#     #
#     # P(gen) / 1 - P(gen) ??
#     # this is Inf for P(gen) = 1.
#     # log of that is logit.
#     # (http://mathworld.wolfram.com/LogitTransformation.html)
#     #
#     # likelihood ratio can also describe tpr / fpr
#     # (http://mathworld.wolfram.com/LikelihoodRatio.html)
#     # but that doesn't make sense for a single sample
#     #
#     if mode == 'step':
#         return 1.0 if dist <= thresh else 0.0
#     # if mode == 'linear':
#     # if mode == 'sigmoid':
#     # if mode == 'logit':
#     else:  # default
#         raise Exception('unknown mode')


# def save_dists(dists, out='distances.txt'):
#     """Save distances to tab delimited txt file. Expects distances as a dict
#        {'Genuine': [...], 'Forged': [...]}, where 'Genuine' contains the
#        target trial llrs."""
#     pass


# ============================================================================


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data_path = sys.argv[1]
    data = pickle.load(open(data_path, 'rb'))
    # data = {'0000': data['0000'], '0000_f': data['0000_f']}

# ============================================================================

    target_dists = genuine_genuine_dists(data, AVG)
    target_dists = np.concatenate([target_dists[k].ravel()
                                   for k in target_dists.keys()])

    nontarget_dists = genuine_forgeries_dists(data, AVG)
    nontarget_dists = np.concatenate([nontarget_dists[k].ravel()
                                      for k in nontarget_dists.keys()])

    max_dist = np.max(np.concatenate((target_dists, nontarget_dists)))
    target_scores = list(map(lambda s: dist_to_score(s, max_dist),
                             target_dists))
    nontarget_scores = list(map(lambda s: dist_to_score(s, max_dist),
                                nontarget_dists))

# ============================================================================

    print_stats(data)

    STEP = 2.0
    thresholds = np.arange(0.0, 150.0, STEP)
    zipped = list(roc(thresholds, data))
    (fpr, tpr, f1, acc, aer) = zip(*zipped)

    best_f1_idx = np.argmax(f1)
    best_acc_idx = np.argmax(acc)
    best_aer_idx = np.argmin(aer)

    print("F1: {:.4f} (threshold = {})".format(f1[best_f1_idx],
                                               thresholds[best_f1_idx]))
    print("Accuracy: {:.4f} (threshold = {})".format(acc[best_acc_idx],
                                                     thresholds[best_acc_idx]))
    print("AER: {:.4f} (threshold = {})".format(aer[best_aer_idx],
                                                thresholds[best_aer_idx]))

# ============================================================================
# F1 score and Accuracy
# ============================================================================

    fig, acc_plot = plt.subplots(2, sharex=True)
    acc_plot[0].plot(thresholds, f1)
    acc_plot[0].set_xlabel('Threshold')
    acc_plot[0].set_ylabel('F1')
    acc_plot[0].set_xticks(np.arange(thresholds[0], thresholds[-1], STEP * 5))
    acc_plot[0].set_title('F1')
    acc_plot[0].set_ylim([0.0, 1.0])
    acc_plot[1].plot(thresholds, acc)
    acc_plot[1].set_xlabel('Threshold')
    acc_plot[1].set_ylabel('Accuracy')
    acc_plot[1].set_xticks(np.arange(thresholds[0], thresholds[-1], STEP * 5))
    acc_plot[1].set_title('Accuracy')
    acc_plot[1].set_ylim([0.0, 1.0])

# ============================================================================
# ROC curve
# ============================================================================

    fig, roc_plot = plt.subplots()
    roc_plot.plot(fpr, tpr)
    roc_plot.set_xlabel('False-Positive-Rate')
    roc_plot.set_ylabel('True-Positive-Rate')
    roc_plot.set_xlim([-0.05, 1.0])
    roc_plot.set_ylim([0.0, 1.05])

    roc_plot.scatter([fpr[best_acc_idx]], [tpr[best_acc_idx]],
                     c='r', edgecolor='r', s=50,
                     label='best accuracy: {:.2%} (t = {})'.format(
        acc[best_acc_idx], thresholds[best_acc_idx]))
    roc_plot.scatter([fpr[best_f1_idx]], [tpr[best_f1_idx]],
                     c='g', edgecolor='g', s=50,
                     label='best F1: {:.2%} (t = {})'.format(
        f1[best_f1_idx], thresholds[best_f1_idx]))
    roc_plot.legend(loc='lower right', scatterpoints=1)


# ============================================================================
# Histograms
# ============================================================================

    HIST_BINS = 70
    fig, hist_plots = plt.subplots(2)
    target_bins, target_bin_edges, patches = \
        hist_plots[0].hist(target_scores, bins=HIST_BINS, range=(0.0, 1.0))
    hist_plots[0].set_title("Histogram target trials")
    nontarget_bins, nontarget_bin_edges, patches = \
        hist_plots[1].hist(nontarget_scores, bins=HIST_BINS, range=(0.0, 1.0))
    hist_plots[1].set_title("Histogram non-target trials")

    data = target_scores
    # data = np.random.poisson(2, 1000)
    fig, poisson_plot = plt.subplots()
    from scipy.misc import factorial
    from scipy.optimize import curve_fit, minimize
    # https://stackoverflow.com/questions/25828184/fitting-to-poisson-histogram#25828558

    def poisson(k, lamb):
        """poisson pdf, parameter lamb is the fit parameter"""
        return (lamb**k/factorial(k)) * np.exp(-lamb)

    # def negLogLikelihood(params, data):
    #     """ the negative log-Likelohood-Function"""
    #     lnl = - np.sum(np.log(poisson(data, params[0])))
    #     return lnl

    # # minimize the negative log-Likelihood
    # result = minimize(negLogLikelihood,  # function to minimize
    #                   x0=np.ones(1),     # start value
    #                   args=(data,),      # additional arguments for function
    #                   method='Powell',   # minimization method, see docs
    #                   )
    # # result is a scipy optimize result object, the fit parameters 
    # # are stored in result.x
    # print(result)

    # plot poisson-deviation with fitted parameter
    x_plot = np.linspace(0, 5, len(data))

    # poisson_plot.hist(data, bins=np.arange(15) - 0.5, normed=True)
    poisson_plot.plot(x_plot, poisson(x_plot, np.mean(data)), 'r-', lw=2)

# ============================================================================
# PDF
# ============================================================================

    # fig, pdf_plot = plt.subplots()
    # NUM_SCORES = np.sum(target_hist[0]) + np.sum(nontarget_hist[0])
    # pdf_plot.plot(target_hist[1][:-1], target_hist[0] / NUM_SCORES, 'g')
    # pdf_plot.plot(nontarget_hist[1][:-1], nontarget_hist[0] / NUM_SCORES, 'r')
    # TODO maybe more interesting on distances histogram

# ============================================================================
# LLR
# ============================================================================

    f = lambda s: llr(s, target_bins[0], target_bin_edges[1],
                      nontarget_bins[0], nontarget_bin_edges[1])

    # TODO: my llr does not work because with my discrete method I have bins in
    # the histogram with 0 samples, resulting in a P(s | trial) of 0.0. This is
    # a problem in both target and nontarget trials: log(0 / ...), log(.../ 0)

# ============================================================================


# ============================================================================
    plt.show()
