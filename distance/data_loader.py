import os
import numpy as np
from scipy.misc import imread


class DataLoader(object):
    """
    A helper class for loading data. Data directories should be organized
    analogously to the gpds synthetic dataset.
    """

    def __init__(self, data_dir, array_module, image_ext='.png'):
        self.data_dir = data_dir
        self.xp = array_module
        self.image_ext = image_ext

    def get_signature_path(self, person, sign_num, variation):
        """Assemble a filename for a signature like 'cf-001-18-05.png' and
           return the full path to the signature image."""
        directory = os.path.join(self.data_dir, "{:03d}".format(person))
        if sign_num > 24:  # a forgery
            prefix = "cf"
            sign_num -= 24
        else:
            prefix = "c"
        fname = "{}-{:03d}-{:02d}-{:02d}{}".format(
            prefix, person, sign_num, variation, self.image_ext)
        return os.path.join(directory, fname)

    def load_image(self, person, sign_num, variation):
        path = self.get_signature_path(person, sign_num, variation)
        return imread(path).astype(self.xp.float32)[self.xp.newaxis, ...]


class TripletLoader(DataLoader):

    def __init__(self, data_dir, array_module, num_classes=4000, skilled_forgeries=False):
        super().__init__(data_dir, array_module)
        # skilled_forgeries parameter indicates whether or not skilled
        # forgeries are allowed to be anchor and positive samples.
        self.num_classes = num_classes
        self.skilled_forgeries = skilled_forgeries

    def get_batch(self, anchor_id, num_triplets):
        """Make a batch using person <anchor_id> as anchor."""
        anchor_samples = list(range(1, 25)) if self.skilled_forgeries else list(range(1, 55))
        np.random.shuffle(anchor_samples)

        # pop anchor_sample, removing it from the remaining anchor_samples
        # the variation of the anchor also needs to be fix  # TODO: why?
        anchor_sign_num = anchor_samples.pop()
        anchor_variation = np.random.randint(1, 21)

        neg_ids = list(range(1, self.num_classes+1))
        neg_ids.remove(anchor_id)
        # allow use of 24 signatures and 30 forgeries of the negatives
        neg_samples = [(np.random.choice(neg_ids),
                        np.random.choice(list(range(1, 55))))
                       for i in range(num_triplets)]

        # for both positive and negative samples we always choose a random variation
        # repeat anchor sample
        a = self.xp.array(
            [self.load_image(anchor_id, anchor_sign_num, anchor_variation)] *
            num_triplets, dtype=self.xp.float32)
        # generate <num_triplets> p's randomly sampled from remaining anchor_samples
        p = self.xp.array(
            [self.load_image(anchor_id, np.random.choice(anchor_samples), np.random.randint(1, 21))
             for _ in range(num_triplets)], dtype=self.xp.float32)
        # negative samples from remaining neg_ids
        n = self.xp.array(
            [self.load_image(np.random.choice(neg_ids), np.random.choice(list(range(1, 55))), np.random.randint(1, 21))
             for _ in range(num_triplets)], dtype=self.xp.float32)
        return self.xp.concatenate([a, p, n])


class LabelDataLoader(DataLoader):

    def get_batch(self, tuples):
        """Return two batches data, labels."""

        data = self.xp.array([self.load_image(user, sign_num, sample)
                              for (user, sign_num, sample) in tuples],
                             dtype=self.xp.float32)
        # NOTE: need to decrement user id for the labels, as
        # softmax_cross_entropy expects the first user to be 0
        labels = self.xp.array([user-1 for (user, _, _) in tuples], dtype=self.xp.int32)
        return data, labels
