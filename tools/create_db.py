import sys
import h5py
import numpy as np
import os
from PIL import Image

from store2hdf5 import store2hdf5, normalize_image, image_to_np_array

TARGET_USER = '1234'


def get_files(no_forgeries=False):
    for (path, _, files) in os.walk("/home/hannes/Data/firmasSINTESISmanuscritas"):
        for f in files:
            if '.jpg' in f and not (no_forgeries and 'cf' in f):
                yield os.path.join(path, f)


def store_files(files, h5file, target_size, chunksize):
    count, num_files = 1, len(files)  # for printing info
    for f in files:
        # label = os.path.basename(os.path.dirname(f))
        label = 1 if TARGET_USER in f else 0
        with Image.open(f) as img:
            print("{}/{} - {}".format(count, num_files, f), end='\r')
            img = img.convert('L')
            np_img = image_to_np_array(img.resize(target_size))
        store2hdf5(h5file, np_img, np.array(label, dtype=np.float32)[np.newaxis, ...], chunksize)
        h5file.flush()
        count += 1


if __name__ == "__main__":
    all_files = list(get_files())
    np.random.shuffle(all_files)
    good = list(filter(lambda x: TARGET_USER in x, all_files))
    bad = list(filter(lambda x: TARGET_USER not in x, all_files))

    # import pdb; pdb.set_trace()

    target_size = (200, 120)

    h5file = h5py.File("data.hdf5", "w")
    for i in range(len(good)-1):
        print("batch {} + {}:{}".format(i, 4000*i, 4000*(i+1)-1))
        batch = [good[i]] + bad[4000*i:4000*(i+1)-1]
        store_files(batch, h5file, target_size, 40)
        print()
    h5file.close()

    # images = np.empty((0, target_height, target_width), dtype=np.float32)
    # labels = np.empty(0, dtype=np.float32)
