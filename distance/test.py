import numpy as np
from scipy.misc import imread, imresize
import chainer
from chainer import optimizers
import os
import pickle

from tripletloss import triplet_loss
from models import EmbedNet

# def test_triplet_loss():
#
#     a = np.array([1,2,3], dtype=np.float32)
#     b = np.array([1,1.5,2.5], dtype=np.float32)
#     c = np.array([3,4,5], dtype=np.float32)
#
#     A, B, C = [chainer.Variable(v, volatile=False) for v in (a,b,c)]
#
#     d = np.array([1,2,3], dtype=np.float32)
#     e = np.array([3,4,5], dtype=np.float32)
#     f = np.array([1,1.5,2.5], dtype=np.float32)
#
#     D, E, F = [chainer.Variable(v, volatile=False) for v in (d,e,f)]
#
#     loss1 = triplet_loss(A,B,C).data
#     loss2 = triplet_loss(D,E,F).data
#
#     return (loss1, loss2)


def get_images(paths):
    imgDir = "/home/hannes/Data/gpdsSynth_scaled/"
    paths = map(lambda p: os.path.join(imgDir, p), paths)
    return np.array([resized(imread(fname))[np.newaxis, ...] for fname in paths],
                    dtype=np.float32)


def resized(img):
    # return img
    return imresize(img, (96, 192))

# def rnd_batches(batchSize, imgHeight, imgWidth):
#     a = np.random.rand(batchSize, 1, imgHeight, imgWidth)
#     p = np.random.rand(batchSize, 1, imgHeight, imgWidth)
#     n = np.random.rand(batchSize, 1, imgHeight, imgWidth)
#     return a, p, n

    # a = get_images(["001/c-001-01.jpg", "003/c-003-01.jpg"])
    # p = get_images(["001/c-001-02.jpg", "004/c-004-01.jpg"])
    # n = get_images(["002/c-002-01.jpg", "003/c-003-02.jpg"])
    # batch = np.concatenate([a, p, n])

    batch = np.random.rand(21, 1, 96, 192).astype(np.float32)


if __name__ == "__main__":
    model = EmbedNet()
    optimizer = optimizers.SGD()
    optimizer.setup(model)

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
