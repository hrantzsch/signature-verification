"""
This script is used to create a training database based on GPDSSynth signatures.
Images are scaled to 192x96;
Paper-like backgrounds are added
"""

import numpy as np
from scipy.misc import imread, imresize, imshow
from PIL import Image

from prepimage import prep


def get_background(bg_files, size):
    img = imread(np.random.choice(bg_files))
    img = np.dot(img[..., :3], [0.299, 0.587, 0.144])
    # import pdb; pdb.set_trace()
    x = np.random.randint(0, img.shape[0]-96)
    y = np.random.randint(0, img.shape[1]-192)
    return imresize(img[y:y+96, x:x+192], (96, 192))


def get_signatures(data_dir, no_forgeries=False):
    for (path, _, files) in os.walk(data_dir):
        for f in files:
            if '.jpg' in f and not (no_forgeries and 'cf' in f):
                yield os.path.join(path, f)


if __name__ == "__main__":
    sig = prep(Image.open("/home/hannes/Data/firmasSINTESISmanuscritas/006/c-006-01.jpg"), (96, 192), 2.0)
    # imshow(sig)
    bg = get_background(["/home/hannes/Downloads/paper/squares_larger.jpg"], None)
    imshow(np.minimum(bg, sig))
