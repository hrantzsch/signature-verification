import numpy as np
from scipy.misc import imread, imresize
import chainer
import os

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


def test_fwd_net():
    model = EmbedNet()

    a = get_images(["001/c-001-01.jpg", "003/c-003-01.jpg"])
    p = get_images(["001/c-001-02.jpg", "004/c-004-01.jpg"])
    n = get_images(["002/c-002-01.jpg", "003/c-003-02.jpg"])
    batch = np.concatenate([a, p, n])
    # import pdb; pdb.set_trace()
    print(model.forward(batch).data)

test_fwd_net()
