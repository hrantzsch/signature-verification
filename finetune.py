import numpy as np

import chainer
from chainer import optimizers
from chainer import links as L
from chainer import serializers

from aux import helpers
from aux.data_loader import LabelDataLoader
from aux.logger import Logger
from aux.train import train
from models.embednet_dnn import DnnWithLinear, DnnComponent

NUM_PERSONAS_PRETRAINED = 40
NUM_PERSONAS = 50

args = helpers.get_args()

if args.gpu >= 0:
    from chainer import cuda
    cuda.check_cuda_available()
    xp = cuda.cupy
else:
    xp = np

dl = LabelDataLoader(args.data, xp, image_ext='.png')
logger = Logger(args.log)

net = DnnWithLinear(NUM_PERSONAS)
model = L.Classifier(net)

if args.gpu >= 0:
    model.to_gpu(args.gpu)

optimizer = optimizers.AdaGrad(lr=0.005)
optimizer.setup(model)

if args.resume:
    print("Warning: resume state given, but ignored for finetuning.")

if args.initmodel:
    print("Using pretrained model " + args.initmodel)
    # TODO init old model like new model above for DnnWithLinear pretrained models
    old_model = L.Classifier(DnnComponent())
    serializers.load_hdf5(args.initmodel, old_model)
    model.predictor.dnn.copyparams(old_model.predictor)
else:
    print("Please provide a pretrained model using -m")
    exit(1)

train_set, test_set = helpers.train_test_tuples(args.test, num_users=NUM_PERSONAS)

train(args.epoch, args.batchsize, optimizer, model, dl,
      train_set, test_set,
      logger, args.out, args.interval)
