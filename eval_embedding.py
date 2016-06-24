"""A script to evaluate a model's embeddings.

The script expects embedded data as a .pkl file.

Currently the script prints min, mean, and max distances intra-class and
comparing a class's samples to the respective forgeries.
"""

# TODO use sqeucledean

import pickle
import sys
import numpy as np
from scipy.spatial.distance import cdist, pdist
from chainer import cuda


DIST_METHOD = 'sqeuclidean'


# ============================================================================


def false_positives(gen_forge_dists, threshold):
    """Number of Genuine-Forgery pairs that are closer than threshold"""
    result = 0
    for user in gen_forge_dists.keys():
        result += np.sum(gen_forge_dists[user] <= threshold)
    return result


def true_negatives(gen_forge_dists, threshold):
    """Number of Genuine-Forgery pairs that are farther than threshold"""
    result = 0
    for user in gen_forge_dists.keys():
        result += np.sum(gen_forge_dists[user] > threshold)
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


def genuine_genuine_dists(data):
    gen_keys = [k for k in data.keys() if 'f' not in k]
    dists = {}
    for k in gen_keys:
        gen = cuda.cupy.asnumpy(data[k])
        dists[k] = cdist(gen, gen, DIST_METHOD)
    return dists


def genuine_forgeries_dists(data):
    gen_keys = [k for k in data.keys() if 'f' not in k]
    dists = {}
    for k in gen_keys:
        gen = cuda.cupy.asnumpy(data[k])
        forge = cuda.cupy.asnumpy(data[k+'_f'])
        np.random.shuffle(forge)
        forge = forge[:4]  # HACK reduce number of forgeries
        dists[k] = cdist(gen, forge, DIST_METHOD)
    return dists


# ============================================================================


def roc(thresholds, data):
    """Compute point in a ROC curve for given thresholds.
       Returns a point (false-pos-rate, true-pos-rate)"""
    gen_forge_dists = genuine_forgeries_dists(data)
    gen_gen_dists = genuine_genuine_dists(data)
    for i in thresholds:
        tp = true_positives(gen_gen_dists, i)
        fp = false_positives(gen_forge_dists, i)
        tn = true_negatives(gen_forge_dists, i)
        fn = false_negatives(gen_gen_dists, i)

        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)

        if tp + fp == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp)
            recall = tpr
            f1 = 2 * precision * recall / (precision + recall)

        acc = (tp + tn) / (tp + fp + fn + tn)

        yield (fpr, tpr, f1, acc)


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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data_path = sys.argv[1]
    data = pickle.load(open(data_path, 'rb'))
    # data = {'0000': data['0000'], '0000_f': data['0000_f']}

    print_stats(data)

    thresholds = range(0, 310, 1)
    zipped = list(roc(thresholds, data))
    (fpr, tpr, f1, acc) = zip(*zipped)

    best_f1_idx = np.argmax(f1)
    best_acc_idx = np.argmax(acc)

    print("F1: {:.4f} (threshold = {})".format(f1[best_f1_idx],
                                               thresholds[best_f1_idx]))
    print("Accuracy: {:.4f} (threshold = {})".format(acc[best_acc_idx],
                                                     thresholds[best_acc_idx]))

    # plt.plot(fpr, tpr)
    # plt.xlabel('False-Positive-Rate')
    # plt.ylabel('True-Positive-Rate')
    # plt.title('ROC curve for threshold {} to {}'.format(thresholds[0],
    #                                                     thresholds[-1]))

    # plt.figure()
    # plt.plot(thresholds, acc)
    # plt.xlabel('Threshold')
    # plt.ylabel('Accuracy')
    # plt.xticks(range(thresholds[0], thresholds[-1], 25))
    # plt.title('Accuracy')

    fig, ax1 = plt.subplots()
    ax1.plot(fpr, tpr)
    ax1.set_xlabel('False-Positive-Rate')
    ax1.set_ylabel('True-Positive-Rate')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])

    ax1.scatter([fpr[best_acc_idx]], [tpr[best_acc_idx]], c='r', edgecolor='r',
                s=50, label='best accuracy: {:.2%} (t = {})'
                            .format(acc[best_acc_idx], best_acc_idx))
    ax1.scatter([fpr[best_f1_idx]], [tpr[best_f1_idx]], c='g', edgecolor='g',
                s=50, label='best F1: {:.2%} (t = {})'
                            .format(f1[best_f1_idx], best_f1_idx))
    ax1.legend(loc='lower right', scatterpoints=1)

    # trying to create a second x axis label for the threshold, but it does not
    # align with the fpr axis...
    # ax2 = ax1.twiny()
    # ax2.set_xlabel('threshold')
    # ax2.set_xticks(range(thresholds[0], thresholds[-1] + 25, 25))
    # ax2.plot(thresholds, tpr)

    plt.show()
