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
        user_result *= (gen_forge_dists[user].shape[0] - (1-AVG)) / \
                        gen_forge_dists[user].shape[1]  # HACK [*1]
        result += user_result
    return result


def true_negatives(gen_forge_dists, threshold):
    """Number of Genuine-Forgery pairs that are farther than threshold"""
    result = 0
    for user in gen_forge_dists.keys():
        user_result = np.sum(gen_forge_dists[user] > threshold)
        user_result *= (gen_forge_dists[user].shape[0] - (1-AVG)) / \
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


def genuine_genuine_dists(data, average=False):
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


def genuine_forgeries_dists(data, average=False):
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
        forge = cuda.cupy.asnumpy(data[k+'_f'])
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
    gen_forge_dists = genuine_forgeries_dists(data)

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data_path = sys.argv[1]
    data = pickle.load(open(data_path, 'rb'))
    # data = {'0000': data['0000'], '0000_f': data['0000_f']}

    print_stats(data)

    thresholds = range(0, 310, 1)
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

    fig, axes = plt.subplots(2, sharex=True)
    axes[0].plot(thresholds, f1)
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('F1')
    axes[0].set_xticks(range(thresholds[0], thresholds[-1], 25))
    axes[0].set_title('F1')
    axes[0].set_ylim([0.0, 1.0])
    axes[1].plot(thresholds, acc)
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xticks(range(thresholds[0], thresholds[-1], 25))
    axes[1].set_title('Accuracy')
    axes[1].set_ylim([0.0, 1.0])

    fig, ax1 = plt.subplots()
    ax1.plot(fpr, tpr)
    ax1.set_xlabel('False-Positive-Rate')
    ax1.set_ylabel('True-Positive-Rate')
    ax1.set_xlim([-0.05, 1.0])
    ax1.set_ylim([0.0, 1.05])

    ax1.scatter([fpr[best_acc_idx]], [tpr[best_acc_idx]], c='r', edgecolor='r',
                s=50, label='best accuracy: {:.2%} (t = {})'
                            .format(acc[best_acc_idx], best_acc_idx))
    ax1.scatter([fpr[best_f1_idx]], [tpr[best_f1_idx]], c='g', edgecolor='g',
                s=50, label='best F1: {:.2%} (t = {})'
                            .format(f1[best_f1_idx], best_f1_idx))
    ax1.legend(loc='lower right', scatterpoints=1)

    plt.show()
