import numpy as np
import time

import chainer
from chainer import cuda
from chainer import optimizers

from tripletembedding.predictors import TripletNet
from tripletembedding.aux import Logger, load_snapshot

from aux.hard_negative_loader import HardNegativeLoader, get_samples, EpochDoneException, NotReadyException
from aux import helpers

from models.vgg_small import VGGSmall


def construct_data_sets(data_dir, skilled, test_ratio):
    samples = list(get_samples(data_dir, skilled))
    np.random.shuffle(samples)
    num_train_samples = int(len(samples) * (1 - test_ratio))
    train_samples = samples[:num_train_samples]
    test_samples = samples[num_train_samples:]
    return train_samples, test_samples


def try_get_batch(dl, model, batchsize):
    try:
        batch = dl.get_batch()
    except NotReadyException:
        dl.prepare_batch(model, batchsize)
        return try_get_batch(dl, model, batchsize)
    return batch


if __name__ == '__main__':
    # time_started = time.time()

    args = helpers.get_args()

    xp = cuda.cupy if args.gpu >= 0 else np
    train, test = construct_data_sets(args.data, False, args.test)
    dl = HardNegativeLoader(cuda.cupy, train, 4 * args.batchsize)

    model = TripletNet(VGGSmall)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model = model.to_gpu()

    optimizer = optimizers.MomentumSGD(lr=0.005)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    if args.initmodel and args.resume:
        load_snapshot(args.initmodel, args.resume, model, optimizer)
        print("Continuing from snapshot. LR: {}".format(optimizer.lr))

    logger = Logger(args, optimizer, args.out)

    dl.prepare_batch(model, args.batchsize)

    for _ in range(1, args.epoch + 1):
        optimizer.new_epoch()
        print('========\nepoch', optimizer.epoch)
        dl.reset()
        # train loop
        while True:
            try:
                batch = try_get_batch(dl, model, args.batchsize)
            except EpochDoneException:
                break
            # concatenate batch to format expected by tripletnet
            batch = cuda.cupy.concatenate(batch, axis=0)

            model.clean()
            x = chainer.Variable(batch)
            optimizer.update(model, x)
            logger.log_iteration("train",
                                 float(model.loss.data), float(model.accuracy),
                                 float(model.mean_diff), float(model.max_diff))
        logger.log_mean("train")
