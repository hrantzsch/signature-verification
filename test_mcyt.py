"""A script to test my model on the MCYT dataset."""

import numpy as np

import chainer
from chainer import cuda
from chainer import optimizers
from chainer import serializers

from aux import helpers
from aux.mcyt_loader import McytLoader
from aux.logger import Logger
from models.tripletnet import TripletNet
from models.alex_dnn import AlexDNN

args = helpers.get_args()

xp = cuda.cupy if args.gpu >= 0 else np
dl = McytLoader(xp)

model = TripletNet(AlexDNN)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    dl.use_device(args.gpu)
    model = model.to_gpu()

print('Loading model from', args.initmodel)
serializers.load_hdf5(args.initmodel, model)

optimizer = optimizers.MomentumSGD(lr=0.0)
optimizer.setup(model)
logger = Logger(args, optimizer, args.out)
test = list(range(100))

for e in range(args.epoch):
    optimizer.new_epoch()
    dl.create_source('test', test, args.batchsize,
                     args.data, skilled=args.skilled)

    for i in range(len(test)):
        x = chainer.Variable(dl.get_batch('test'))
        loss = model(x)
        logger.log_iteration("test", float(model.loss.data),
                             float(model.accuracy), float(model.dist))
logger.log_mean("test")
