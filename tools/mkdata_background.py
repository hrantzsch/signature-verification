"""
This script is used to create a training database based on GPDSSynth signatures.
Images are scaled to 192x96;
Paper-like backgrounds are added
"""

import argparse
import numpy as np
from scipy.misc import imread, imresize, imshow, imsave
from PIL import Image
import os
from skimage.transform import rotate
import time

import prepimage


def load_backgrounds(folder):
    """read image file and convert to grayscale"""
    return [np.dot(imread(os.path.join(folder, bg_file))[..., :3], [0.299, 0.587, 0.144])
            for bg_file in os.listdir(folder) if '.jpg' in bg_file or '.png' in bg_file]


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


def process_signature(sig_path):
    sig = imread(sig_path).astype(np.float32) / 255.0
    sig = rotate(sig, np.random.randint(-25, 25), cval=1.0, resize=True)

    roiy, roix = get_roi(sig)
    shape = (roiy[1] - roiy[0], roix[1] - roix[0])
    bg = get_background(np.random.choice(backgrounds), shape).astype(np.float32) / 255.0

    img = bg + sig[roiy[0]:roiy[1], roix[0]:roix[1]]
    img = imresize(img, target_size, mode='L').astype(np.float32)
    img *= 1.0/img.max()
    # return np.minimum(img, 1.0)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('signatures',
                        help='Path to extracted GPDS data')
    parser.add_argument('backgrounds',
                        help='Path to background files (jpg or png)')
    parser.add_argument('--out', '-o', default='images',
                        help='Path to save output images')
    parser.add_argument('--start', '-s', default=1, type=int,
                        help='User to start with (for resumes)')
    args = parser.parse_args()

    target_size = (96, 192)
    # signatures = list(get_signatures(args.signatures))
    backgrounds = load_backgrounds(args.backgrounds)
    print("Loaded {} backgrounds".format(len(backgrounds)))

    for user in range(args.start, 4001):
        user_str = "{:03d}".format(user)
        print("processing user " + user_str)
        os.makedirs(os.path.join(args.out, user_str), exist_ok=True)
        count = 0
        start = time.clock()
        for sig in get_signatures(os.path.join(args.signatures, user_str)):
            fname, _ = os.path.splitext(os.path.basename(sig))
            for i in range(1, 21):
                outname = os.path.join(args.out, user_str, "{}-{:02d}.png".format(fname, i))
                imsave(outname, process_signature(sig), 'png')
                count += 1
        print("{} images in {:3f} sec".format(count, time.clock() - start))
