import numpy as np
from skimage.filters import threshold_otsu
import skimage.transform as tf
from scipy.misc import imresize, imsave, fromimage, imshow, imread
from scipy.ndimage import interpolation as ip
import os


def make_folders(root, target_root):
    for d in os.listdir(root):
        abs_path = os.path.join(root, d)
        if os.path.isdir(abs_path):
            os.makedirs(os.path.join(target_root, d), exist_ok=True)
            make_folders(abs_path, os.path.join(target_root, d))


def binarize(image):
    thresh = threshold_otsu(image)
    binary = image <= thresh
    return binary


def normalize(image, new_min, new_max):
    return (image - np.min(image)) *\
           (new_max - new_min) /\
           (np.max(image) - np.min(image)) +\
           new_min


def crop(img):
    binary = binarize(img)
    cols = np.where(np.sum(binary, axis=1))
    rows = np.where(np.sum(binary, axis=0))
    left, right = np.min(cols), np.max(cols)
    top, bottom = np.min(rows), np.max(rows)
    return img[left:right, top:bottom]


def augment(filename, target_size, angle, distortion):
    # read image
    img = imread(filename, mode='L')

    # whiten noise
    img[img > 220.0] = 255.0

    img = ip.rotate(img, angle, cval=255.0)

    # perspective transform
    distortion *= img.shape[0]
    d1 = abs(distortion)
    d2 = 0

    # from left or right depending on sign
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
    pad = 3
    img = imresize(crop(img), (target_size[0] - 2*pad, target_size[1] - 2*pad))
    img = np.pad(img, pad, mode='constant', constant_values=255.0)

    img = normalize(img, 0, 1)

    return img


def get_files(data_dir, pattern=""):
    """Get all files that include <pattern>"""
    for (path, _, files) in os.walk(data_dir):
        for f in files:
            if ('.png' in f.lower() or '.jpg' in f.lower()) and \
               pattern in f:
                yield os.path.join(path, f)


if __name__ == "__main__":
    # export SCIPY_PIL_IMAGE_VIEWER=feh
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('out')
    parser.add_argument('dataset',
                        help="Name of the dataset. Supported sets: "
                        "gpdss, mcyt")
    args = parser.parse_args()

    if args.dataset == "gpdss":
        g_pattern = 'c-'
        f_pattern = 'cf-'
    elif args.dataset == "mcyt":
        g_pattern = 'v'
        f_pattern = 'f'
    else:
        print("Error: unsupported dataset. Supported dataset: gpdss, mcyt")
        exit()

    imgDir = args.data

    genuines = list(get_files(imgDir, g_pattern))
    forgeries = list(get_files(imgDir, f_pattern))
    num_imgs = len(genuines) + len(forgeries)

    g_target = os.path.join(args.out, 'Genuine')
    f_target = os.path.join(args.out, 'Forged')
    os.makedirs(g_target, exist_ok=True)
    make_folders(args.data, g_target)
    os.makedirs(f_target, exist_ok=True)
    make_folders(args.data, f_target)

    count = 0
    distortions = [-0.3, 0, 0.3]
    rotations = [-45, -30, -15, 0, 15, 30, 45]

    for (base, target) in [(genuines, g_target), (forgeries, f_target)]:
        for f in base:
            num_aug = 0
            for dist in distortions:
                for rot in rotations:
                    rot += np.random.randint(-3, 4)

                    print("{:4d}/{:4d} - {}".format(count, num_imgs, f),
                          end='\r')

                    image = augment(f, (96, 192), rot, dist)

                    # fname: outdir/persona/sample-name_num-aug.png
                    sname = os.path.splitext(os.path.relpath(f, imgDir))[0]
                    fname = "{}_{:02d}.png".format(sname, num_aug)
                    fname = os.path.join(target, fname)
                    num_aug += 1

                    imsave(fname, image)
            count += 1
