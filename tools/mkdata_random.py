import argparse
import h5py
import numpy as np
import os
from PIL import Image

from store2hdf5 import store2hdf5, normalize_image, image_to_np_array
from prepimage import prep


def get_files(data_dir, no_forgeries=False):
    for (path, _, files) in os.walk(data_dir):
        for f in files:
            if '.jpg' in f and not (no_forgeries and 'cf' in f):
                yield os.path.join(path, f)


def store_files(files, h5file, chunksize, prep_image=True):
    if prep_image:
        print("Images will be scaled and cropped!")
    count, num_files = 1, len(files)  # for printing info
    for f in files:
        label = os.path.basename(os.path.dirname(f))
        with Image.open(f) as img:
            print("{}/{} - {}".format(count, num_files, f), end='\r')
            if prep_image:
                img = prep(img, (100, 200), (2.0 / 1))
            np_img = image_to_np_array(img)
        store2hdf5(h5file, np_img, np.array(label, dtype=np.float32)[np.newaxis, ...], chunksize)
        h5file.flush()
        count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data',
                        help='Path to extracted GPDS data')
    parser.add_argument('out', default='data.hdf5',
                        help='Path to save hdf5 data')
    parser.add_argument('--forgeries', '-f', default=False, type=bool,
                        help='Label skilled forgeries as positive sample')
    parser.add_argument('--prepare', '-p', default=False, type=bool,
                        help='Crop and resize images')
    args = parser.parse_args()

    all_files = list(get_files(args.data, no_forgeries=not args.forgeries))
    np.random.shuffle(all_files)

    h5file = h5py.File(args.out, "w")
    store_files(all_files, h5file, 20, prep_image=args.prepare)
    h5file.close()
