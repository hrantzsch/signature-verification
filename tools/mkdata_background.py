"""
This script is used to create a training database based on GPDSSynth signatures.
Images are scaled to 192x96;
Paper-like backgrounds are added
"""

import numpy as np
from scipy.misc import imread, imresize, imshow, imsave
from PIL import Image
import os

import prepimage


def load_backgrounds(folder="/home/hannes/Data/paper_bg"):
    """read image file and convert to grayscale"""
    return [np.dot(imread(os.path.join(folder, bg_file))[..., :3], [0.299, 0.587, 0.144])
            for bg_file in os.listdir(folder) if '.jpg' in bg_file]


def get_background(img, size):
    """crop a random piece of desired size from the given image"""
    y = np.random.randint(0, img.shape[0]-size[0])
    x = np.random.randint(0, img.shape[1]-size[1])
    return imresize(img[y:y+size[0], x:x+size[1]], (size[0], size[1]))


def get_signatures(data_dir, no_forgeries=False):
    for (path, _, files) in os.walk(data_dir):
        for f in files:
            if '.jpg' in f and not (no_forgeries and 'cf' in f):
                yield os.path.join(path, f)


def get_roi(image, pad=20):
    roix, roiy = prepimage.min_max(prepimage.binarize(image))
    roix = (max(0, roix[0] - pad), min(roix[1] + pad, image.shape[1]))
    roiy = (max(0, roiy[0] - pad), min(roiy[1] + pad, image.shape[0]))
    return roiy, roix


if __name__ == "__main__":
    target_size = (96, 192)
    signatures = list(get_signatures("/home/hannes/Data/firmasSINTESISmanuscritas/"))
    backgrounds = load_backgrounds()

    for i in range(50):
        sig_file = np.random.choice(signatures)
        sig = imread(sig_file)
        roiy, roix = get_roi(sig)
        shape = (roiy[1] - roiy[0], roix[1] - roix[0])
        bg = get_background(np.random.choice(backgrounds), shape)

        img = np.minimum(bg, sig[roiy[0]:roiy[1], roix[0]:roix[1]])
        img = imresize(img, target_size, mode='L')
        imsave("/home/hannes/tmp/images/{}.png".format(os.path.basename(sig_file)), img)
