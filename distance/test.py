import numpy as np
from scipy.misc import imread
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


def test_fwd_net():
    model = EmbedNet()

    imgDir = "/home/hannes/Data/gpdsSynth_scaled/"
    a = np.array([imread(fname)[np.newaxis, ...] for fname in [
        os.path.join(imgDir, "001/c-001-01.jpg"),
        os.path.join(imgDir, "003/c-003-01.jpg"),
    ]], dtype=np.float32)
    p = np.array([imread(fname)[np.newaxis, ...] for fname in [
        os.path.join(imgDir, "001/c-001-02.jpg"),
        os.path.join(imgDir, "004/c-004-01.jpg"),
    ]], dtype=np.float32)
    n = np.array([imread(fname)[np.newaxis, ...] for fname in [
        os.path.join(imgDir, "002/c-002-01.jpg"),
        os.path.join(imgDir, "003/c-003-02.jpg"),
    ]], dtype=np.float32)
    batch = np.concatenate([a, p, n])
    import pdb; pdb.set_trace()
    model.forward(batch)

test_fwd_net()
