import chainer
from chainer import serializers


class Logger:

    def __init__(self, log_file=None):
        self.iteration = 0
        self.sum_loss = 0
        self.sum_acc = 0
        self.log_file = log_file
        self.current_section = ''

    def make_snapshot(self, model, optimizer, epoch, name):
        serializers.save_hdf5('{}_{}.model'.format(name, epoch), model)
        serializers.save_hdf5('{}_{}.state'.format(name, epoch), optimizer)
        print("Snapshot created")

    def load_snapshot(self, model_path, state_path, model, optimizer):
        print('Load model from', model_path)
        serializers.load_hdf5(model_path, model)
        print('Load optimizer state from', state_path)
        serializers.load_hdf5(state_path, optimizer)

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
                f.write(label)
                f.write('\n')
                self.current_section = label
            f.write("{},{},{}\n".format(self.iteration, loss, acc))
