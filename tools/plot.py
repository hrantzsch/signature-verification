import argparse
import matplotlib.pyplot as plt
from itertools import chain


def parse(logfile):
    loss_train = []
    loss_test = []
    acc_train = []
    acc_test = []
    section_loss = []
    section_acc = []
    with open(args.logfile, 'r') as logfile:
        for line in logfile:
            if line.startswith('#'):  # skip comments, e.g. the header
                continue
            if 'train' in line:
                if len(section_loss) > 0:
                    loss_test.append(section_loss)
                    section_loss = []
                if len(section_acc) > 0:
                    acc_test.append(section_acc)
                    section_acc = []
            elif 'test' in line:
                if len(section_loss) > 0:
                    loss_train.append(section_loss)
                    section_loss = []
                if len(section_acc) > 0:
                    acc_train.append(section_acc)
                    section_acc = []
            else:
                section_loss.append(float(line.split(',')[1]))
                if 'None' not in line:
                    section_acc.append(float(line.split(',')[2]))
    if len(section_loss) > 0:
        loss_test.append(section_loss)
    if len(section_acc) > 0:
        acc_test.append(section_acc)

    return loss_train, loss_test, acc_train, acc_test


def avg(l):
    return sum(l) / len(l)


def plot_avg(logfile):
    loss_train, loss_test, acc_train, acc_test = parse(logfile)

    f, axarr = plt.subplots(2, sharex=True)
    x = list(range(1, len(loss_train)+1))
    # x = list(range(len(loss_train)))  # old style starting at epoch 0

    axarr[0].plot(x, list(map(avg, loss_train)), '.-', label='train')
    axarr[0].plot(x, list(map(avg, loss_test)), '.-', label='test')
    # axarr[0].set_ylim([0.45, 0.55])
    axarr[0].set_title("loss")
    axarr[0].legend(loc='upper right')
    if len(acc_train) > 0:
        axarr[1].plot(x, list(map(avg, acc_train)), '.-')
    axarr[1].plot(x, list(map(avg, acc_test)), 'g.-')
    axarr[1].set_title("acc")
    # axarr[1].set_ylim([0.2, 0.3])

    plt.show()


def plot_test(logfile):
    _, loss, _, acc = parse(logfile)

    plt.plot(list(map(avg, loss)), label='loss')
    plt.plot(list(map(avg, acc)), label='acc')
    plt.legend(loc='upper left')
    plt.show()


def plot_mixed(logfile):
    loss_train, loss_test, acc_train, acc_test = parse(logfile)

    f, axarr = plt.subplots(2, sharex=True)
    x = list(range(1, len(loss_train)+1))
    # x = list(range(len(loss_train)))  # old style starting at epoch 0

    loss_chain = list(chain.from_iterable(loss_train))
    axarr[0].plot(loss_chain, '-')
    axarr[0].plot(list(range(1, len(loss_chain)+1, len(loss_chain)//len(loss_train))),
                  list(map(avg, loss_test)), '.-')
    acc_chain = list(chain.from_iterable(acc_train))
    axarr[1].plot(acc_chain, '-')
    axarr[1].plot(list(range(1, len(acc_chain)+1, len(acc_chain)//len(acc_train))),
                  list(map(avg, acc_test)), '.-')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile')
    # parser.add_argument('--style', default='avg',
    #                     help="'avg' for average, or 'all'")
    args = parser.parse_args()

    plot_avg(args.logfile)
