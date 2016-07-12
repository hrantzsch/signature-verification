"""A script to evaluate a model's embeddings.

The script expects embedded data as a .pkl file.

Currently the script prints min, mean, and max distances intra-class and
comparing a class's samples to the respective forgeries.
"""

import pickle
import subprocess
import sys
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy import stats
from scipy.optimize import fmin
from scipy.special import gamma as gammaf
from chainer import cuda

import seaborn as sns
sns.set_palette('colorblind')


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

        yield (fpr, fnr, tpr, f1, acc, aer)


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
    return max(0, 1 - dist / max_dist)


def llr(s, func, target_params, nontarget_params):
    """Compute the log-likelihood ratio.
       `s` is either a distance or a score computed from that distance.
       `target_params` and `nontarget_params` are the parameters to `func`,
       either for the curve fitted to the (non)target distances or scores.
    """
    # TODO: llr(score, target_score_distr, nontarget_score_distr) !=
    #       llr(dist,  target_dist_distr,  nontarget_dist_distr)
    return np.log(func(s, *target_params) / func(s, *nontarget_params))


# def neglogsigmoid(x):
#     return -np.log(1 / (1 + np.exp(-x)))


# def cllr(target_llrs, nontarget_llrs):
#     # % target trials
#     # c1 = mean(neglogsigmoid(tar_llrs))/log(2);
#     # % non_target trials
#     # c2 = mean(neglogsigmoid(-nontar_llrs))/log(2);
#     # cllr = (c1+c2)/2;
#     c1 = np.mean(map(neglogsigmoid, target_llrs)) / np.log(2)
#     c2 = np.mean(map(neglogsigmoid, nontarget_llrs)) / np.log(2)
#     return (c1+c2)/2

# ============================================================================

def write_score(out, target_dists, nontarget_dists,
                func, target_params, nontarget_params):
    """Write a score file as expected by the FoCal toolkit[1].
       [1]: TODO... sorry, not now
    """
    with open(out, 'w') as f:
        for d in target_dists:
            f.write("1 {} {}\n".format(func(d, *target_params),
                                       func(d, *nontarget_params)))
        for d in nontarget_dists:
            f.write("2 {} {}\n".format(func(d, *target_params),
                                       func(d, *nontarget_params)))

# ============================================================================


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.scale as mscale
    from tools.ProbitScale import ProbitScale

    mscale.register_scale(ProbitScale)

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

    HIST_BINS = 350
    target_bins, target_bin_edges = np.histogram(target_scores,
                                                 bins=HIST_BINS,
                                                 range=(0.0, 1.0),
                                                 density=True)
    nontarget_bins, nontarget_bin_edges = np.histogram(nontarget_scores,
                                                       bins=HIST_BINS,
                                                       range=(0.0, 1.0),
                                                       density=True)

    target_dbins, target_dbin_edges = np.histogram(target_dists,
                                                   bins=HIST_BINS,
                                                   range=(0.0, max_dist),
                                                   density=True)
    nontarget_dbins, nontarget_dbin_edges = np.histogram(nontarget_dists,
                                                         bins=HIST_BINS,
                                                         range=(0.0, max_dist),
                                                         density=True)

# ============================================================================

    print_stats(data)

    STEP = 10.0
    thresholds = np.arange(0.0, max_dist + STEP, STEP)
    zipped = list(roc(thresholds, data))
    (fpr, fnr, tpr, f1, acc, aer) = zip(*zipped)

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

    fig, dhist_plots = plt.subplots(2)

    width = 1.0 * (target_dbin_edges[1] - target_dbin_edges[0])
    center = (target_dbin_edges[:-1] + target_dbin_edges[1:]) / 2
    dhist_plots[0].bar(center, target_dbins, align='center', width=width)

    width = 1.0 * (nontarget_dbin_edges[1] - nontarget_dbin_edges[0])
    center = (nontarget_dbin_edges[:-1] + nontarget_dbin_edges[1:]) / 2
    dhist_plots[1].bar(center, nontarget_dbins, align='center', width=width)

    dhist_plots[0].set_title("Distance histogram target trials")
    dhist_plots[1].set_title("Distance histogram non-target trials")

    fig, hist_plots = plt.subplots(2)

    width = 1.0 * (target_bin_edges[1] - target_bin_edges[0])
    center = (target_bin_edges[:-1] + target_bin_edges[1:]) / 2
    hist_plots[0].bar(center, target_bins, align='center', width=width)

    width = 1.0 * (nontarget_bin_edges[1] - nontarget_bin_edges[0])
    center = (nontarget_bin_edges[:-1] + nontarget_bin_edges[1:]) / 2
    hist_plots[1].bar(center, nontarget_bins, align='center', width=width)

    hist_plots[0].set_title("Score histogram target trials")
    hist_plots[1].set_title("Score histogram non-target trials")

# ============================================================================
# Weibull distribution fit
# ============================================================================

    # fit weibull to scores
    xx = np.linspace(0, 1.0, 500)

    target_wb = stats.exponweib.fit(target_scores, 1, 1, scale=2, loc=0)
    yy = stats.exponweib.pdf(xx, *target_wb)
    hist_plots[0].plot(xx, yy)

    nontarget_wb = stats.exponweib.fit(nontarget_scores, 1, 1, scale=2, loc=0)
    yy = stats.exponweib.pdf(xx, *nontarget_wb)
    hist_plots[1].plot(xx, yy)

    # fig, wbs = plt.subplots()
    # wbs.plot(xx, stats.exponweib.cdf(xx, *target_wb), 'g')
    # wbs.plot(xx, stats.exponweib.pdf(xx, *target_wb), 'g')
    # wbs.plot(xx, stats.exponweib.cdf(xx, *nontarget_wb), 'r')
    # wbs.plot(xx, stats.exponweib.pdf(xx, *nontarget_wb), 'r')

    # fit weibull to distances

    xx = np.linspace(0, max_dist, 500)

    target_d_wb = stats.exponweib.fit(target_dists, 1, 1, scale=2, loc=0)
    yy = stats.exponweib.pdf(xx, *target_d_wb)
    dhist_plots[0].plot(xx, yy)

    nontarget_d_wb = stats.exponweib.fit(nontarget_dists, 1, 1, scale=2, loc=0)
    yy = stats.exponweib.pdf(xx, *nontarget_d_wb)
    dhist_plots[1].plot(xx, yy)

# ============================================================================
# Beta distribution fit -- doesn't work that well
# ============================================================================

    # target_beta = stats.beta.fit(target_scores)
    # nontarget_beta = stats.beta.fit(nontarget_scores)
    # hist_plots[0].plot(xx, stats.beta.pdf(xx, *target_beta), 'g')
    # hist_plots[1].plot(xx, stats.beta.pdf(xx, *nontarget_beta), 'g')

# ============================================================================
# PDF
# ============================================================================

    # fig, pdf_plot = plt.subplots()
    # NUM_SCORES = np.sum(target_hist[0]) + np.sum(nontarget_hist[0])
    # pdf_plot.plot(target_hist[1][:-1], target_hist[0] / NUM_SCORES, 'g')
    # pdf_plot.plot(nontarget_hist[1][:-1], nontarget_hist[0] / NUM_SCORES, 'r')
    # TODO maybe more interesting on distances histogram

# ============================================================================
# LLR computation
# ============================================================================

    # target_llrs = list(map(lambda d: llr(d, stats.exponweib.pdf,
    #                                      target_d_wb, nontarget_d_wb),
    #                        target_dists))
    # nontarget_llrs = list(map(lambda d: llr(d, stats.exponweib.pdf,
    #                                         target_d_wb, nontarget_d_wb),
    #                           nontarget_dists))


# ============================================================================
# DET-Plot
# ============================================================================

    fig, det_plot = plt.subplots()
    plt.xscale('probit')
    plt.yscale('probit')
    plt.xlim([0, 0.5])
    plt.ylim([0, 0.5])
    det_plot.set_title("DET-plot")
    det_plot.plot(fpr, fnr)
    det_plot.plot(plt.xticks()[0], plt.yticks()[0], ':')

# ============================================================================
# FoCal Toolkit
# ============================================================================

    # score_out = "out.score"
    # # write_score(score_out, target_scores, nontarget_scores,
    # #             stats.exponweib.pdf, target_wb, nontarget_wb)
    # write_score(score_out, target_dists, nontarget_dists,
    #             stats.exponweib.pdf, target_d_wb, nontarget_d_wb)
    # print("Wrote score to", score_out)

    # cmd = "java -jar /home/hannes/src/cllr_evaluation/jFocal/VectorCal.jar " \
    #       "-analyze -t ./out.score"
    # subprocess.run(cmd.split())

# ============================================================================
# Plot
# ============================================================================

    plt.show()
