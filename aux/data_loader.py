import os
import numpy as np
from scipy.misc import imread

import queue
import threading

from chainer import cuda


QUEUE_SIZE = 8  # maybe make it an argument later


def get_signature_path(person, sign_num, variation, data_dir, extension):
    """Assemble a filename for a signature like 'cf-001-18-05.png' and
       return the full path to the signature image."""
    directory = os.path.join(data_dir, "{:03d}".format(person))
    if sign_num > 24:  # a forgery
        prefix = "cf"
        sign_num -= 24
    else:
        prefix = "c"
    fname = "{}-{:03d}-{:02d}-{:02d}{}".format(
        prefix, person, sign_num, variation, extension)
    return os.path.join(directory, fname)


def load_image(person, sign_num, variation,
               array_module, data_dir, extension='.png'):
    path = get_signature_path(person, sign_num, variation, data_dir, extension)
    return imread(path).astype(array_module.float32)[array_module.newaxis, ...]


class DataLoader(object):
    """A helper class for loading data. Data directories should be organized
    analogously to the gpds synthetic dataset.
    """

    def __init__(self, data_dir, array_module, image_ext='.png'):
        self.data_dir = data_dir
        self.xp = array_module
        self.image_ext = image_ext

        self.queue = queue.Queue(QUEUE_SIZE)

    def prepare_triplet_provider(self, anchors, num_triplets, num_classes):
        """Prepare to deliver data for one epoch.
        I.e. initiate loading data into the queue and ensure that data is
        available
        """
        NUM_WORKERS = 1
        for i in range(NUM_WORKERS):
            anchors_part = anchors[i::NUM_WORKERS]
            if not self.queue.empty():
                print("Warning: queue not empty on prepare_epoch")
            self.data_provider = TripletLoader(
                anchors_part, self.queue, self.data_dir, self.xp,
                num_triplets, num_classes, False, self.image_ext, cuda.Device())
            self.data_provider.start()

    def get_batch(self):
        return self.queue.get()


# class LabelDataLoader(DataProvider):
#
#     def load_batch(self, tuples):
#         """Return two batches data, labels."""
#         # TODO change to accept one tuple
#         data = self.xp.array([self.load_image(user, sign_num, sample)
#                               for (user, sign_num, sample) in tuples],
#                              dtype=self.xp.float32)
#         # NOTE: need to decrement user id for the labels, as
#         # softmax_cross_entropy expects the first user to be 0
#         labels = self.xp.array([user-1 for (user, _, _) in tuples], dtype=self.xp.int32)
#         return data, labels


class TripletLoader(threading.Thread):

    def __init__(self, anchors, queue, data_dir, xp, num_triplets,
                 num_classes, skilled_forgeries, image_ext, device):
        # skilled_forgeries parameter indicates whether or not skilled
        # forgeries are allowed to be anchor and positive samples.
        threading.Thread.__init__(self)
        self.anchors = anchors
        self.data_dir = data_dir
        self.xp = xp
        self.image_ext = image_ext
        self.num_triplets = num_triplets
        self.num_classes = num_classes
        self.num_variations = 20
        self.skilled_forgeries = skilled_forgeries
        self.queue = queue
        self.device = device

    def run(self):
        for a in self.anchors:
            data = self.load_batch(a)
            self.queue.put(data)  # blocking, no timeout

    def load_batch(self, anchor_id):
        self.device.use()
        """Make a batch using person <anchor_id> as anchor."""
        anchor_samples = list(range(1, 25)) if self.skilled_forgeries else list(range(1, 55))
        np.random.shuffle(anchor_samples)

        # pop anchor_sample, removing it from the remaining anchor_samples
        # the variation of the anchor also needs to be fix  # TODO: why?
        anchor_sign_num = anchor_samples.pop()
        anchor_variation = np.random.randint(1, self.num_variations+1)

        neg_ids = list(range(1, self.num_classes+1))
        neg_ids.remove(anchor_id)
        # allow use of 24 signatures and 30 forgeries of the negatives
        neg_samples = [(np.random.choice(neg_ids),
                        np.random.choice(list(range(1, 55))))
                       for i in range(self.num_triplets)]

        # for both positive and negative samples we always choose a random variation
        # repeat anchor sample
        a = self.xp.array(
            [load_image(anchor_id, anchor_sign_num, anchor_variation, self.xp, self.data_dir)] *
            self.num_triplets, dtype=self.xp.float32)
        # generate <num_triplets> p's randomly sampled from remaining anchor_samples
        p = self.xp.array(
            [load_image(anchor_id, np.random.choice(anchor_samples), np.random.randint(1, self.num_variations+1), self.xp, self.data_dir)
             for _ in range(self.num_triplets)], dtype=self.xp.float32)
        # negative samples from remaining neg_ids
        n = self.xp.array(
            [load_image(np.random.choice(neg_ids), np.random.choice(list(range(1, 55))), np.random.randint(1, self.num_variations+1), self.xp, self.data_dir)
             for _ in range(self.num_triplets)], dtype=self.xp.float32)

        return self.xp.concatenate([a, p, n])
