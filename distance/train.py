import numpy as np
from scipy.misc import imread, imresize
import chainer
from chainer import optimizers
from chainer import cuda
import os
import pickle

from tripletloss import triplet_loss
from models import EmbedNet


data_dir = "/home/hannes/Data/gpdsSynth_200x100/"


def get_signatures(person):
    """Return the paths to all authentic signatures of a person"""
    directory = os.path.join(data_dir, "{:03d}".format(person))
    for f in os.scandir(directory):
        if 'cf' not in f.name:
            yield f.path


def get_signature_path(person, sign_num):
    directory = os.path.join(data_dir, "{:03d}".format(person))
    if sign_num > 24:  # a forgery
        prefix = "cf"
        sign_num -= 24
    else:
        prefix = "c"
    fname = "{}-{:03d}-{:02d}.jpg".format(prefix, person, sign_num)
    return os.path.join(directory, fname)


def load_triplet(anc, pos, neg):
    """Load the requested triplet of images as a numpy array.
       anc, pos, and neg are expected to be tuples of (person, sign_number)
    """
    return [imread(get_signature_path(num, sign)).astype(np.float32)[np.newaxis, ...]
                     for (num, sign) in (anc, pos, neg)]

#
# def get_batch(anchor):
#     """Make a batch using person <anchor> as anchor.
#        Batchsize is fixed to 70 triplets."""
#     anchor_samples = list(get_signatures(anchor))
#     np.random.shuffle(anchor_samples)
#     # one anchor
#     a = anchor_samples.pop()
#     # 70 positives (includes duplicates)
#     p = np.random.choice(anchor_samples, 70)
#     negatives = list(range(1, 4001))
#     negatives.remove(anchor)
#     np.random.shuffle(negatives)
#     n = []
#     while len(n) < 70:
#         n.append(np.random.choice(list(get_signatures(negatives.pop()))))
#     import pdb; pdb.set_trace()


def get_batch(anchor_id):
    """Make a batch using person <anchor> as anchor.
       Batchsize is fixed to 70 triplets."""
    anchor_samples = list(range(1, 25))
    np.random.shuffle(anchor_samples)
    anchor_sample = anchor_samples.pop()

    pos_samples = np.random.choice(anchor_samples, 70)

    neg_ids = list(range(1, 4001))
    neg_ids.remove(anchor_id)
    # allow use of 24 signatures and 30 forgeries of the negatives
    neg_samples = [(np.random.choice(neg_ids),
                    np.random.choice(list(range(1, 55))))
                   for i in range(70)]

    a = np.array([imread(get_signature_path(anchor_id, anchor_sample)).astype(np.float32)[np.newaxis, ...]] * 70, dtype=np.float32)
    p = np.array([imread(get_signature_path(anchor_id, pos_samples[i])).astype(np.float32)[np.newaxis, ...] for i in range(70)], dtype=np.float32)
    n = np.array([imread(get_signature_path(neg_samples[i][0], neg_samples[i][1])).astype(np.float32)[np.newaxis, ...] for i in range(70)], dtype=np.float32)
    return np.concatenate([a, p, n])


def get_images(paths):
    imgDir = "/home/hannes/Data/gpdsSynth_scaled/"
    paths = map(lambda p: os.path.join(imgDir, p), paths)
    return np.array([resized(imread(fname))[np.newaxis, ...] for fname in paths],
                    dtype=np.float32)


if __name__ == "__main__":
    # args

    # numpy or cuda arrays
    # xp = cuda.cupy

    # model setup
    model = EmbedNet()
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    batch = get_batch(5)
    print(model.forward(batch).data)
    exit(0)

    for epoch in range(1, n_epoch + 1):
        print('epoch', epoch)

        # training
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        for i in range(0, N_train, batchsize):
            x_batch = np.asarray(data[perm[i:i + batchsize]])
            y_batch = np.asarray(labels[perm[i:i + batchsize]])

            optimizer.zero_grads()
            loss, acc = forward(x_batch, y_batch)
            print("Iteration {}: Loss {}; Acc {}".format(i, float(loss.data), float(acc.data)))
            loss.backward()
            optimizer.update()

            if epoch == 1 and i == 0:
                with open("graph.dot", "w") as o:
                    o.write(c.build_computational_graph((loss, )).dump())
                with open("graph.wo_split.dot", "w") as o:
                    g = c.build_computational_graph((loss, ),
                                                    remove_split=True)
                    o.write(g.dump())
                print('graph generated')

            sum_loss += float(loss.data) * len(y_batch)
            sum_accuracy += float(acc.data) * len(y_batch)

        print('train mean loss={}, accuracy={}'.format(
            sum_loss / N, sum_accuracy / N))

        # evaluation
        sum_accuracy = 0
        sum_loss = 0
        for i in range(N_test, N, batchsize):
            # x_batch = xp.asarray(data[i:i + batchsize])
            # y_batch = xp.asarray(labels[i:i + batchsize])
            x_batch = np.asarray(data[i:i+batchsize])
            y_batch = np.asarray(labels[i:i+batchsize])

            loss, acc = forward(x_batch, y_batch, train=False)

            sum_loss += float(loss.data) * len(y_batch)
            sum_accuracy += float(acc.data) * len(y_batch)

        print('test  mean loss={}, accuracy={}'.format(
            sum_loss / N_test, sum_accuracy / N_test))
