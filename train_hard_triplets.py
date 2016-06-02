import numpy as np
import os               # get_samples
import time             # time stats

import chainer
from chainer import cuda
from chainer import optimizers

from tripletembedding.predictors import TripletNet
from tripletembedding.aux import Logger, load_snapshot

from aux.triplet_loader import TripletLoader
from aux.hard_negative_loader import HardNegativeLoader
from aux.hard_negative_loader import EpochDoneException, NotReadyException
from aux import helpers

from models.vgg_small_legacy import VGGSmall


# def construct_data_sets(data_dir, skilled, test_ratio):
#     samples = list(get_samples(data_dir, skilled))
#     np.random.shuffle(samples)
#     num_train_samples = int(len(samples) * (1 - test_ratio))
#     train_samples = samples[:num_train_samples]
#     test_samples = samples[num_train_samples:]
#     return train_samples, test_samples


def get_samples(data_dir, skilled, anchors=[]):
    """Returns a generator on lists of files per class in directory.
       skilled indicates whether to include skilled forgeries.
       If anchors is not [], only use classes in anchors.
       anchors must be a list of int."""
    skilled_condition = lambda f: (skilled and 'f' in f) or 'f' not in f
    anchor_condition = lambda d: anchors == [] or int(d) in anchors
    for d in os.listdir(data_dir):
        path = os.path.join(data_dir, d)
        if not (os.path.isdir(path) and anchor_condition(d)):
            continue
        files = os.listdir(path)
        for f in files:
            if '.png' in f and skilled_condition(f):
                yield (d, os.path.join(path, f))


def try_get_batch(dl, model, batchsize):
    try:
        batch = dl.get_batch()
    except NotReadyException:
        dl.prepare_batch(model, batchsize)
        return try_get_batch(dl, model, batchsize)
    return batch


if __name__ == '__main__':
    args = helpers.get_args()

    xp = cuda.cupy if args.gpu >= 0 else np
    train_anchors, test_anchors = helpers.train_test_anchors(args.test, 4000)
    train = list(get_samples(args.data, False, train_anchors))

    print("Training set size: {}".format(len(train)))

    dl = HardNegativeLoader(cuda.cupy, train, int(2.5 * args.batchsize))
    dl_test = TripletLoader(xp)

    model = TripletNet(VGGSmall)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model = model.to_gpu()

    optimizer = optimizers.MomentumSGD(lr=0.001)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    # without clipping, gradients for a pretrained model become too large
    optimizer.add_hook(chainer.optimizer.GradientClipping(10.0))

    logger = Logger(args, optimizer, args.out)

    if args.initmodel and args.resume:
        load_snapshot(args.initmodel, args.resume, model, optimizer)
        print("Continuing from snapshot. LR: {}".format(optimizer.lr))
        # testing
        # dl_test.create_source('test', test_anchors, args.batchsize,
        #                       args.data, skilled=args.skilled)

        # for i in range(len(test_anchors)):
        #     x = chainer.Variable(dl_test.get_batch('test'), volatile=True)
        #     loss = model(x)
        #     logger.log_iteration("test",
        #                          float(model.loss.data), float(model.accuracy),
        #                          float(model.mean_diff), float(model.max_diff))
        # logger.log_mean("test")

    for _ in range(1, args.epoch + 1):
        t_iteration = []
        t_get_batch = []
        t_update = []

        optimizer.new_epoch()
        print("==== epoch {} ===".format(optimizer.epoch))
        dl.reset()
        # train loop
        while True:
            t_it_started = time.time()
            try:
                batch = try_get_batch(dl, model, args.batchsize)
            except EpochDoneException:
                break
            t_get_batch.append(time.time() - t_it_started)
            # concatenate batch to format expected by tripletnet
            batch = cuda.cupy.concatenate(batch, axis=0)

            model.clean()
            x = chainer.Variable(batch)
            t_up_started = time.time()
            optimizer.update(model, x)

            t_update.append(time.time() - t_up_started)
            t_iteration.append(time.time() - t_it_started)

            logger.log_iteration("train",
                                 float(model.loss.data), float(model.accuracy),
                                 float(model.mean_diff), float(model.max_diff))

        print()
        logger.log_mean("train")
        print("mean iteration time: {:.2f} sec".format(np.mean(t_iteration)))
        print("mean batch sampling time: {:.2f} sec".format(np.mean(t_get_batch)))
        print("mean update time: {:.2f} sec".format(np.mean(t_update)))

        if optimizer.epoch % args.lrinterval == 0 and optimizer.lr > 0.000001:
            optimizer.lr *= 0.5
            logger.mark_lr()
            print("learning rate decreased to {}".format(optimizer.lr))
            if optimizer.epoch % args.interval == 0:
                logger.make_snapshot(model)

        # testing
        dl_test.create_source('test', test_anchors, args.batchsize,
                              args.data, skilled=args.skilled)

        for i in range(len(test_anchors)):
            x = chainer.Variable(dl_test.get_batch('test'), volatile=True)
            loss = model(x)
            logger.log_iteration("test",
                                 float(model.loss.data), float(model.accuracy),
                                 float(model.mean_diff), float(model.max_diff))
        logger.log_mean("test")
