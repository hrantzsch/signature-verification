import numpy as np
from skimage.filters import threshold_otsu
from scipy.misc import imresize, imsave, fromimage
from PIL import Image


def _find_min(image, row=0):
    if np.any(image[row]):
        return row
    else:
        return _find_min(image, row+1)


def _find_max(image, row):
    if np.any(image[row]):
        return row
    else:
        return _find_max(image, row-1)


def min_max(image):
    y_min = _find_min(image)
    y_max = _find_max(image, image.shape[0]-1)
    img_rot = np.rot90(image)
    x_min = _find_min(img_rot)
    x_max = _find_max(img_rot, img_rot.shape[0]-1)
    return (x_min, x_max), (y_min, y_max)


def crop(image, ratio):
    thresh = threshold_otsu(image)
    binary = image <= thresh

    (x_min, x_max), (y_min, y_max) = min_max(binary)
    src_w, src_h = x_max-x_min, y_max-y_min

    if (src_w / src_h) > ratio:  # pad height
        pad = (1/ratio) * src_w - src_h
        y_min -= pad / 2
        y_max += pad / 2
    elif (src_w / src_h) < ratio:
        pad = ratio * src_h - src_w
        x_min -= pad / 2
        x_max += pad / 2

    image = np.pad(image, 500, mode='edge')  # yes, it's ugly
    return image[y_min+500:y_max+500, x_min+500:x_max+500]


if __name__ == "__main__":
    from mkdata_random import get_files
    import os

    images = list(get_files("/home/hannes/Data/firmasSINTESISmanuscritas"))
    num_imgs = len(images)

    image = images[0]

    count = 0
    for f in images:
        with Image.open(f) as image:
            print("{}/{} - {}".format(count, num_imgs, f), end='\r')
            count += 1
            image = crop(fromimage(image), (2.0 / 1))
            image = imresize(image, (100, 200), mode='L')
            imsave("/home/hannes/tmp/images/{}".format(os.path.basename(f)), image)


    # print (avg_x, avg_y, avg_x / num_imgs, avg_y / num_imgs)
    # 86551127 168976381 400.69966203703706 782.2980601851851