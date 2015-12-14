import numpy as np

import chainer
from chainer import optimizers
from chainer import links as L

import aux
from train import train
from models.dnn import DnnWithLinear
from data_loader import LabelDataLoader
from logger import Logger


args = aux.get_args()

NUM_USERS = 10

if args.gpu >= 0:
    from chainer import cuda
    cuda.check_cuda_available()
    xp = cuda.cupy
else:
    xp = np

dl = LabelDataLoader(args.data, xp, image_ext='.png')
logger = Logger(args.log)

net = DnnWithLinear(NUM_USERS)
model = L.Classifier(net)

if args.gpu >= 0:
    model.to_gpu(args.gpu)

optimizer = optimizers.AdaGrad(lr=0.005)
optimizer.setup(model)

if args.initmodel and args.resume:
    logger.load_snapshot(args.initmodel, args.resume, model, optimizer)

train_set, test_set = aux.train_test_tuples(args.test, num_users=NUM_USERS)

train(args.epoch, args.batchsize, optimizer, model, dl,
      train_set, test_set,
      logger, args.out, args.interval)
