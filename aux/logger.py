import sys
from datetime import date, datetime

import chainer
from chainer import serializers


def load_snapshot(model_path, state_path, model, optimizer):
    print('Load model from', model_path)
    serializers.load_hdf5(model_path, model)
    print('Load optimizer state from', state_path)
    serializers.load_hdf5(state_path, optimizer)


class Logger:

    def __init__(self, args, optimizer, name, extra_msg=''):
        self.iteration = 0
        self.sum_loss = 0
        self.sum_acc = 0
        self.current_section = ''
        self.optimizer = optimizer

        # setup according to arguments
        self.name = name if name is not '' else 'signdist'
        self.out_file = "{}_{}".format(date.isoformat(date.today()), self.name)
        self.log_file = "{}.log".format(self.out_file)
        # write config to head of the log file
        self.write_config(args, extra_msg)

    def make_snapshot(self, model):
        # TODO: get model from Optimizer
        prefix = "{}_{}".format(self.out_file, self.optimizer.epoch)
        serializers.save_hdf5(prefix + ".model", model)
        serializers.save_hdf5(prefix + ".state", self.optimizer)
        print("Snapshot created")

    def log_iteration(self, label, loss, acc=None):
        self.iteration += 1

        print("{} {:04d}:\tloss={:.4f}".format(label, self.iteration, loss),
              end='' if acc is not None else '\r')
        self.sum_loss += loss
        if acc is not None:
            print(", acc={:.3%}".format(acc), end='\r')
            self.sum_acc += acc

        if self.log_file is not None:
            self.write_iteration(label, loss, acc)

    def log_mean(self, label):
        print("{} mean\tloss={:.4f}".format(label, self.sum_loss / self.iteration),
              end='' if self.sum_acc > 0 else '\n')
        if self.sum_acc > 0:
            print(", acc={:.3%}".format(self.sum_acc / self.iteration))
        self.iteration = 0
        self.sum_loss = 0
        self.sum_acc = 0

    def write_iteration(self, label, loss, acc):
        with open(self.log_file, 'a+') as f:
            if self.current_section != label:
                f.write("{} [{}]".format(label, self.optimizer.epoch))
                f.write('\n')
                self.current_section = label
            f.write("{},{},{}\n".format(self.iteration, loss, acc))

    def write_config(self, args, extra_msg):
        with open(self.log_file, 'a+') as f:
            self._comment("=" * 40, f)
            self._comment("{} initiated at {}".format(
                self.name, datetime.isoformat(datetime.now())
            ), f)
            self._comment("-" * 40, f)  # arguments passed
            self._comment("Data: " + args.data, f)
            self._comment("Batchsize: {}".format(args.batchsize), f)
            self._comment("Test ratio: {}".format(args.test), f)
            self._comment("Hard triplet ratio: {}".format(args.skilled), f)
            dev = "CPU" if args.gpu < 0 else "GPU ".format(args.gpu)
            self._comment("Device: " + dev, f)
            if args.initmodel:
                self._comment("Init model: " + args.initmodel, f)
            if args.resume:
                self._comment("Resume state: " + args.resume, f)
            self._comment("-" * 40, f)  # parameters set in script
            self._comment("Optimizer: " + self.optimizer.__class__.__name__, f)
            self._comment("Epoch: {}".format(self.optimizer.epoch), f)
            if extra_msg:
                self._comment(extra_msg, f)
            self._comment("-" * 40, f)  # complete call
            self._comment("{}".format(sys.argv), f)
            self._comment("=" * 40, f)

    def _comment(self, msg, f):
        f.write("# {}\n".format(msg))
