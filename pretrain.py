import numpy as np

import chainer
from chainer import optimizers
from chainer import links as L

from aux import helpers
from aux.data_loader import LabelDataLoader
from aux.logger import Logger
from aux.train import train
from models.embednet_dnn import DnnWithLinear


args = helpers.get_args()

NUM_USERS = 200

if args.gpu >= 0:
    from chainer import cuda
    cuda.check_cuda_available()
    xp = cuda.cupy
    cuda.get_device(1).use()
    print("using gpu", args.gpu)
else:
    xp = np

dl = LabelDataLoader(args.data, xp, image_ext='.png')
logger = Logger(args.log)

net = DnnWithLinear(200)
model = L.Classifier(net)

if args.gpu >= 0:
    model.to_gpu(args.gpu)

optimizer = optimizers.AdaGrad(lr=0.005)
optimizer.setup(model)

if args.initmodel and args.resume:
    logger.load_snapshot(args.initmodel, args.resume, model, optimizer)

train_set, test_set = helpers.train_test_tuples(args.test, num_users=NUM_USERS)

train(args.epoch, args.batchsize, optimizer, model, dl,
      train_set, test_set,
      logger, args.out, args.interval)
