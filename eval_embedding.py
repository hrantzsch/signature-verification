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


def logit(x):
    return np.log(x / (1 - x))


def dist_to_score(dist, max_dist=500):
    """Supposed to compute P(target_trial | s)"""
    return max(1 / max_dist, 1 - dist / max_dist)


def llr(dist, condition_positive_rate):
    """Computes log-Likelihood-ratio given a sample's distance and the ratio of
       condition positive samples in the data."""
    # TODO provide a max_dist or find a more intelligent way
    return logit(dist_to_score(dist)) - logit(condition_positive_rate)


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


def save_dists(dists, out='distances.txt'):
    """Save distances to tab delimited txt file. Expects distances as a dict
       {'Genuine': [...], 'Forged': [...]}, where 'Genuine' contains the
       target trial llrs."""
    pass


# ============================================================================


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data_path = sys.argv[1]
    data = pickle.load(open(data_path, 'rb'))
    # data = {'0000': data['0000'], '0000_f': data['0000_f']}

# ============================================================================

    dists_target = genuine_genuine_dists(data, AVG)
    dists_target = np.concatenate([dists_target[k].ravel()
                                   for k in dists_target.keys()])

    dists_nontarget = genuine_forgeries_dists(data, AVG)
    dists_nontarget = np.concatenate([dists_nontarget[k].ravel()
                                      for k in dists_nontarget.keys()])

    max_dist = np.max(np.concatenate((dists_target, dists_nontarget)))
    scores_target = list(map(lambda s: dist_to_score(s, max_dist),
                             dists_target))
    scores_nontarget = list(map(lambda s: dist_to_score(s, max_dist),
                            dists_nontarget))

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

    fig, hist_plots = plt.subplots(2)
    hist_plots[0].hist(scores_target, bins='auto', range=(0.0, 1.0))
    hist_plots[0].set_title("Histogram target trials")
    hist_plots[1].hist(scores_nontarget, bins='auto', range=(0.0, 1.0))
    hist_plots[1].set_title("Histogram non-target trials")

# ============================================================================

    plt.show()
