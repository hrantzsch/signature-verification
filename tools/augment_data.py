import numpy as np
from skimage.filters import threshold_otsu
import skimage.transform as tf
from scipy.misc import imresize, imsave, fromimage, imshow, imread
import os


def binarize(image):
    thresh = threshold_otsu(image)
    binary = image <= thresh
    return binary


def crop(img):
    binary = binarize(img)
    cols = np.where(np.sum(binary, axis=1))
    rows = np.where(np.sum(binary, axis=0))
    left, right = np.min(cols), np.max(cols)
    top, bottom = np.min(rows), np.max(rows)
    return img[left:right, top:bottom]


def augment(filename, target_size, angle, distortion):
    # read image
    img = imread(filename, mode='L') / 255.0
    img = tf.rotate(img, angle, cval=1.0)

    # perspective transform
    distortion *= img.shape[0]
    d1 = abs(distortion)
    d2 = 0

    # perspective from left or right at random
    if distortion < 0:
        d1, d2 = d2, d1
    src = np.array((
        (0, d1),
        (0, img.shape[0] - d1),
        (img.shape[1], img.shape[0] - d2),
        (img.shape[1], d2),
        ))

    dst = np.array((
        (0, 0),
        (0, img.shape[0]),
        (img.shape[1], img.shape[0]),
        (img.shape[1], 0),
        ))

    tform = tf.ProjectiveTransform()
    tform.estimate(src, dst)
    img = tf.warp(img, tform, output_shape=img.shape, cval=1.0)

    # crop, resize, pad
    pad = 5
    img = imresize(crop(img), (target_size[0] - pad, target_size[1] - pad))
    img = np.pad(img, pad, mode='constant', constant_values=255.0)

    return img


def get_files(data_dir, no_forgeries=False):
    for (path, _, files) in os.walk(data_dir):
        for f in files:
            if '.png' in f.lower():
                yield os.path.join(path, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('out')
    args = parser.parse_args()

    imgDir = args.data
    outDir = args.out

    images = list(get_files(imgDir))
    num_imgs = len(images)

    for i in range(0, 100):
        os.makedirs(os.path.join(outDir, "{:04d}".format(i)), exist_ok=True)

    count = 0
    distortions = [-0.21, -0.14, -0.07, 0, 0.07, 0.14, 0.21]
    rotations = [-14, -7, 0, 7, 14]
    for f in images:
        num_aug = 0
        for d in distortions:
            for rot in rotations:
                rot += np.random.randint(-3, 4)

                print("{:4d}/{:4d} - {}".format(count, num_imgs, f), end='\r')

                image = augment(f, (96, 192), rot, d)

                # fname: outdir/persona/sample-name_num-aug.png
                sample_name = os.path.splitext(os.path.relpath(f, imgDir))[0]
                fname = "{}_{:02d}.png".format(sample_name, num_aug)
                fname = os.path.join(outDir, fname)
                num_aug += 1

                imsave(fname, image)
        count += 1
