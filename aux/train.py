import numpy as np
import chainer

import aux


def train(epochs, batchsize,
          optimizer, lossfun, data_loader,
          train_set, test_set,
          logger, out_name, snapshot_interval):
    graph_generated = False
    for _ in range(1, epochs + 1):
        optimizer.new_epoch()
        print('epoch', optimizer.epoch)

        # training
        np.random.shuffle(train_set)
        for i in range(0, len(train_set), batchsize):
            x_data, t_data = data_loader.get_batch(train_set[i:i+batchsize])
            x = chainer.Variable(x_data)
            t = chainer.Variable(t_data)

            optimizer.update(lossfun, x, t)
            logger.log_iteration("train", float(lossfun.loss.data),
                                 float(lossfun.accuracy.data))

            if not graph_generated:
                aux.write_graph(lossfun.loss)
                graph_generated = True

        logger.log_mean("train")

        if optimizer.epoch % 15 == 0 and optimizer.lr > 0.0002:
            optimizer.lr *= 0.5
            print("learning rate decreased to {}".format(optimizer.lr))
        if optimizer.epoch % snapshot_interval == 0:
            logger.make_snapshot(lossfun, optimizer, optimizer.epoch, out_name)

        # testing
        np.random.shuffle(test_set)
        for i in range(0, len(test_set), batchsize):
            x_data, t_data = data_loader.get_batch(test_set[i:i+batchsize])
            x = chainer.Variable(x_data)
            t = chainer.Variable(t_data)
            loss = lossfun(x, t)
            logger.log_iteration("test", float(lossfun.loss.data),
                                 float(lossfun.accuracy.data))
        logger.log_mean("test")
