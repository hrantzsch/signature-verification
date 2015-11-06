import argparse
import h5py
import numpy as np
import os
from PIL import Image

from store2hdf5 import store2hdf5, normalize_image, image_to_np_array

parser = argparse.ArgumentParser()
parser.add_argument('data',
                    help='Path to extracted GPDS data')
parser.add_argument('out', default='data.hdf5',
                    help='Path to save hdf5 data')
parser.add_argument('--forgeries', '-f', default=False,
                    help='Label skilled forgeries as positive sample')
args = parser.parse_args()


def get_files(no_forgeries=False):
    for (path, _, files) in os.walk(args.data):
        for f in files:
            if '.jpg' in f and not (no_forgeries and 'cf' in f):
                yield os.path.join(path, f)


def store_files(files, h5file, target_size, chunksize):
    count, num_files = 1, len(files)  # for printing info
    for f in files:
        label = os.path.basename(os.path.dirname(f))
        with Image.open(f) as img:
            print("{}/{} - {}".format(count, num_files, f), end='\r')
            img = img.convert('L')
            np_img = image_to_np_array(img.resize(target_size))
        store2hdf5(h5file, np_img, np.array(label, dtype=np.float32)[np.newaxis, ...], chunksize)
        h5file.flush()
        count += 1


if __name__ == "__main__":
    all_files = list(get_files(no_forgeries=not args.forgeries))
    np.random.shuffle(all_files)

    target_size = (200, 120)

    h5file = h5py.File(args.out, "w")
    store_files(all_files, h5file, target_size, 20)
    h5file.close()
