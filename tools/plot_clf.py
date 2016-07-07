import argparse
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("colorblind")
sns.set_color_codes("colorblind")


def parse(logfile):
    loss_train = []
    loss_test = []
    section_loss = []
    acc_train = []
    acc_test = []
    section_acc = []
    mean_diff_train = []
    mean_diff_test = []
    section_mean_diff = []
    max_diff_train = []
    max_diff_test = []
    section_max_diff = []
    with open(logfile, 'r') as lf:
        for line in lf:
            if line.startswith('#') or line == '\n':  # skip comments and blank
                continue
            if 'train' in line:
                if len(section_loss) > 0:
                    loss_test.append(section_loss)
                    section_loss = []
                if len(section_acc) > 0:
                    acc_test.append(section_acc)
                    section_acc = []
                if len(section_mean_diff) > 0:
                    mean_diff_test.append(section_mean_diff)
                    section_mean_diff = []
                if len(section_max_diff) > 0:
                    max_diff_test.append(section_max_diff)
                    section_max_diff = []
            elif 'test' in line:
                if len(section_loss) > 0:
                    loss_train.append(section_loss)
                    section_loss = []
                if len(section_acc) > 0:
                    acc_train.append(section_acc)
                    section_acc = []
                if len(section_mean_diff) > 0:
                    mean_diff_train.append(section_mean_diff)
                    section_mean_diff = []
                if len(section_max_diff) > 0:
                    max_diff_train.append(section_max_diff)
                    section_max_diff = []
            else:
                # it, loss, acc, mean_diff = line.split(','); max_diff = None
                it, loss, acc, mean_diff, max_diff = line.split(',')
                section_loss.append(float(loss))
                if acc is not None:
                    section_acc.append(float(acc))
                if mean_diff is not None:
                    section_mean_diff.append(float(mean_diff))
                if max_diff is not None:
                    section_max_diff.append(float(max_diff))
    if len(section_loss) > 0:
        loss_test.append(section_loss)
    if len(section_acc) > 0:
        acc_test.append(section_acc)
    if len(section_mean_diff) > 0:
        mean_diff_test.append(section_mean_diff)
    if len(section_max_diff) > 0:
        max_diff_test.append(section_max_diff)

    return loss_train, loss_test, acc_train, acc_test,\
        mean_diff_train, mean_diff_test, max_diff_train, max_diff_test


def avg(l):
    return sum(l) / len(l)


def plot_avg(logfile):
    loss_train, loss_test, acc_train, acc_test, mean_diff_train,\
        mean_diff_test, max_diff_train, max_diff_test = parse(logfile)

    f, axarr = plt.subplots(2, sharex=True)
    x = list(range(1, len(loss_train)+1))
    # x = list(range(len(loss_train)))  # old style starting at epoch 0

    axarr[0].plot(x, list(map(avg, loss_train)), '.-', label='train')
    axarr[0].plot(x, list(map(avg, loss_test)), '.-', label='test')
    # axarr[0].set_ylim([-0.05, 5.0])
    axarr[0].set_title("loss")
    axarr[0].legend(loc='upper right')

    axarr[1].plot(x, list(map(avg, acc_train)), '.-')
    if len(acc_train) > 0:
        axarr[1].plot(x, list(map(avg, acc_test)), '.-')
        # axarr[1].set_ylim([0.0, 1.1])
    axarr[1].set_title("accuracy")

    # if len(mean_diff_train) > 0:
    #     axarr[2].plot(x, list(map(avg, mean_diff_train)), '.-')
    #     if len(mean_diff_test) > 0:
    #         axarr[2].plot(x, list(map(avg, mean_diff_test)), 'g.-')
    #     axarr[2].set_title("mean_diff")
    #
    # if len(max_diff_train) > 0:
    #     axarr[3].plot(x, list(map(avg, max_diff_train)), '.-')
    #     if len(max_diff_test) > 0:
    #         axarr[3].plot(x, list(map(avg, max_diff_test)), 'g.-')
    #     axarr[3].set_title("max_diff")
    # axarr[1].set_ylim([0.7, 1.0])

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile')
    # parser.add_argument('--style', default='avg',
    #                     help="'avg' for average, or 'all'")
    args = parser.parse_args()

    plot_avg(args.logfile)

# def plot_test(logfile):
#     _, loss, _, acc = parse(logfile)
#
#     plt.plot(list(map(avg, loss)), label='loss')
#     plt.plot(list(map(avg, acc)), label='acc')
#     plt.legend(loc='upper left')
#     plt.show()
#
#
# def plot_mixed(logfile):
#     loss_train, loss_test, acc_train, acc_test = parse(logfile)
#
#     f, axarr = plt.subplots(2, sharex=True)
#     x = list(range(1, len(loss_train)+1))
#     # x = list(range(len(loss_train)))  # old style starting at epoch 0
#
#     loss_chain = list(chain.from_iterable(loss_train))
#     axarr[0].plot(loss_chain, '-')
#     axarr[0].plot(list(range(1, len(loss_chain)+1, len(loss_chain)//len(loss_train))),
#                   list(map(avg, loss_test)), '.-')
#     acc_chain = list(chain.from_iterable(acc_train))
#     axarr[1].plot(acc_chain, '-')
#     axarr[1].plot(list(range(1, len(acc_chain)+1, len(acc_chain)//len(acc_train))),
#                   list(map(avg, acc_test)), '.-')
#     plt.legend(loc='lower right')
#     plt.show()
