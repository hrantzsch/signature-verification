import os
import numpy as np
from scipy.misc import imread
import pickle

from chainer import optimizers
from chainer import cuda
from chainer import computational_graph as c

from tripletloss import triplet_loss
from models import EmbedNet


data_dir = "/extra/data_sets/GPDSSyntheticSignatures/gpds_96x192/"


def get_signature_path(person, sign_num):
    directory = os.path.join(data_dir, "{:03d}".format(person))
    if sign_num > 24:  # a forgery
        prefix = "cf"
        sign_num -= 24
    else:
        prefix = "c"
    fname = "{}-{:03d}-{:02d}.jpg".format(prefix, person, sign_num)
    return os.path.join(directory, fname)


def get_batch(anchor_id, num_triplets):
    """Make a batch using person <anchor_id> as anchor."""
    anchor_samples = list(range(1, 25))
    np.random.shuffle(anchor_samples)
    anchor_sample = anchor_samples.pop()

    pos_samples = np.random.choice(anchor_samples, num_triplets)

    neg_ids = list(range(1, 4001))
    neg_ids.remove(anchor_id)
    # allow use of 24 signatures and 30 forgeries of the negatives
    neg_samples = [(np.random.choice(neg_ids),
                    np.random.choice(list(range(1, 55))))
                   for i in range(num_triplets)]

    a = np.array([imread(get_signature_path(anchor_id, anchor_sample)).astype(np.float32)[np.newaxis, ...]] * num_triplets, dtype=np.float32)
    p = np.array([imread(get_signature_path(anchor_id, pos_samples[i])).astype(np.float32)[np.newaxis, ...] for i in range(num_triplets)], dtype=np.float32)
    n = np.array([imread(get_signature_path(neg_samples[i][0], neg_samples[i][1])).astype(np.float32)[np.newaxis, ...] for i in range(num_triplets)], dtype=np.float32)
    return np.concatenate([a, p, n])


if __name__ == "__main__":
    # args
    n_epoch = 10
    batch_triplets = 3  # batchsize will be 3 * batch_triplets

    # model setup
    model = EmbedNet()
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    graph_generated = False

    for epoch in range(1, n_epoch + 1):
        print('epoch', epoch)

        # training
        anchors = list(range(1, 4001))
        np.random.shuffle(anchors)

        sum_loss = 0
        iteration = 0
        for i in anchors:
            iteration += 1
            x_batch = get_batch(i, batch_triplets)

            optimizer.zero_grads()
            loss = model.forward(x_batch)
            print("Iteration {}: Loss {}".format(iteration, float(loss.data)))
            loss.backward()
            optimizer.update()

            if not graph_generated:
                with open("graph.dot", "w") as o:
                    o.write(c.build_computational_graph((loss, )).dump())
                with open("graph.wo_split.dot", "w") as o:
                    g = c.build_computational_graph((loss, ),
                                                    remove_split=True)
                    o.write(g.dump())
                graph_generated = True
                print('graph generated')

            sum_loss += float(loss.data) * len(x_batch)

        print('train mean loss={}'.format(sum_loss / N))

        # evaluation -- later...
        # sum_accuracy = 0
        # sum_loss = 0
        # for i in range(N_test, N, batchsize):
        #     # x_batch = xp.asarray(data[i:i + batchsize])
        #     # y_batch = xp.asarray(labels[i:i + batchsize])
        #     x_batch = np.asarray(data[i:i+batchsize])
        #     y_batch = np.asarray(labels[i:i+batchsize])
        #
        #     loss, acc = forward(x_batch, y_batch, train=False)
        #
        #     sum_loss += float(loss.data) * len(y_batch)
        #     sum_accuracy += float(acc.data) * len(y_batch)
        #
        # print('test  mean loss={}, accuracy={}'.format(
        #     sum_loss / N_test, sum_accuracy / N_test))
