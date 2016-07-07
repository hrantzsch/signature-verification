"""A script to train a triplet distance based model.

This script is used to train models that expect to be fed triplets for
training. Currently it can be used to train EmbedNet, based on the FaceNet
paper, and TripletNet, based on Hoffer, Ailon: "Deep Metric Learning Using
Triplet Network".
"""

import numpy as np
import time

import chainer
from chainer import cuda
from chainer import optimizers

from tripletembedding.predictors import TripletNet
from tripletembedding.aux import Logger, load_snapshot

from aux import helpers
from aux.triplet_loader import TripletLoader
from aux.mcyt_loader import McytLoader
from aux.index_loader import IndexLoader, load_dict, anchors_in

from models.vgg_small import VGGSmall, VGGSmallConv, VGGClf
from models.vgg_xs import VGGXSClf, VGGXSEmbed
# import models.vgg_small_legacy as legacy
import queue

import pickle


args = helpers.get_args()

xp = cuda.cupy if args.gpu >= 0 else np
dl = IndexLoader(xp)

data = load_dict(args.data)

# === inter-class split ===

# - loading existing train/test split for continued training -
# train_anchors, test_anchors = pickle.load(open('snapshots/16-06-20_vgg_small_triplets/vgg_small_triplets_train_test_split.pkl', 'rb'))
# - create train and test dictionaries from train/test split -
# train = {
#     'Genuine': {label: data['Genuine'][label]
#                 for label in train_anchors['Genuine'].keys()},
#     'Forged': {label: data['Forged'][label]
#                for label in train_anchors['Genuine'].keys()}
# }
# test = {
#     'Genuine': {label: data['Genuine'][label]
#                 for label in test_anchors['Genuine'].keys()},
#     'Forged': {label: data['Forged'][label]
#                for label in test_anchors['Genuine'].keys()}
# }

# - create new inter-class split -
# all_ancs = list(data['Genuine'].keys())
# np.random.shuffle(all_ancs)
# t = int(len(all_ancs) * args.test)
# train_anchors, test_anchors = all_ancs[:-t], all_ancs[-t:]
# # - create train and test dictionaries from train/test split -
# train = {
#     'Genuine': {label: data['Genuine'][label] for label in train_anchors},
#     'Forged': {label: data['Forged'][label] for label in train_anchors}
# }
# test = {
#     'Genuine': {label: data['Genuine'][label] for label in test_anchors},
#     'Forged': {label: data['Forged'][label] for label in test_anchors}
# }

# === intra-class split -- writer-dependent ===
train = {group: {} for group in data.keys()}
test = {group: {} for group in data.keys()}
for group in data.keys():
    for label in data[group].keys():
        samples = data[group][label]
        np.random.shuffle(samples)

        t = int(len(samples) * args.test)
        train[group][label] = samples[:-t]
        test[group][label] = samples[-t:]

# - save train/test split -
pickle.dump([train, test], open(args.out+'_train_test_split.pkl', 'wb'))

model = TripletNet(VGGSmall)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    dl.use_device(args.gpu)
    model = model.to_gpu()

optimizer = optimizers.MomentumSGD(lr=0.01)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
optimizer.add_hook(chainer.optimizer.GradientClipping(10.0))

if args.initmodel and args.resume:
    load_snapshot(args.initmodel, args.resume, model, optimizer)
    print("Continuing from snapshot. LR: {}".format(optimizer.lr))

# load pre-trained CNN
# from chainer import serializers
# pretrained = VGGClf(100)
# serializers.load_hdf5(args.initmodel, pretrained)
# model.cnn.conv.copyparams(pretrained.conv)

logger = Logger(args, optimizer, args.out)

graph_generated = False
for _ in range(1, args.epoch + 1):
    time_started = time.time()

    optimizer.new_epoch()
    print('========\nepoch', optimizer.epoch)

    # margin = min(3.0, 1.0 + 0.005 * optimizer.epoch**2)
    # print('margin:\t{}'.format(margin))
    margin = 1.0

    # training
    dl.create_source('train',
                     args.batchsize, train, skilled=args.skilled)
    dl.create_source('test',
                     args.batchsize, test, skilled=args.skilled)

    while True:
        model.zerograds()
        model.clean()

        try:
            x_data = dl.get_batch('train')
        except queue.Empty:
            break

        x = chainer.Variable(x_data)
        optimizer.update(model, x)
        # optimizer.update(model, x, margin)
        logger.log_iteration("train",
                             float(model.loss.data), float(model.accuracy),
                             float(model.mean_diff), float(model.max_diff))

        if not graph_generated:
            helpers.write_graph(model.loss)
            graph_generated = True

    logger.log_mean("train")
    print("iteration time:\t{:.3f} sec"
          .format((time.time() - time_started) / len(train)))

    if optimizer.epoch % args.lrinterval == 0 and optimizer.lr > 0.000001:
        optimizer.lr *= 0.5
        logger.mark_lr()
        print("learning rate decreased to {}".format(optimizer.lr))
    if optimizer.epoch % args.interval == 0:
        logger.make_snapshot(model)

    # testing
    while True:
        try:
            x = chainer.Variable(dl.get_batch('test'), volatile=True)
        except queue.Empty:
            break
        loss = model(x)
        logger.log_iteration("test",
                             float(model.loss.data), float(model.accuracy),
                             float(model.mean_diff), float(model.max_diff))
    logger.log_mean("test")

# make final snapshot if not just taken one
if optimizer.epoch % args.interval != 0:
    logger.make_snapshot(model)
