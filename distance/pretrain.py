import numpy as np

import chainer
from chainer import optimizers
from chainer import links as L

import aux
from train import train
from models.dnn import DnnWithLinear
from data_loader import LabelDataLoader
from logger import Logger


def train_test_set(test_fraction, num_users):
    sign_per_user = 54
    sample_per_sign = 20
    t = int(test_fraction * num_users * sign_per_user * sample_per_sign)
    data = [(user, sign, sample)
            for user in range(1, num_users + 1)
            for sign in range(1, sign_per_user + 1)
            for sample in range(1, sample_per_sign + 1)]
    np.random.shuffle(data)
    return data[:-t], data[-t:]

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

train_set, test_set = train_test_set(args.test, num_users=NUM_USERS)

train(args.epoch, args.batchsize, optimizer, model, dl,
      train_set, test_set,
      logger, args.out, args.interval)
