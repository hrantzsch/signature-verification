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
    print((x_min, x_max), (y_min,y_max))
    return (x_min, x_max), (y_min,y_max)


def crop(image, ratio):
    dest_w, dest_h = ratio[0], ratio[1]

    thresh = threshold_otsu(image)
    binary = np.asarray(image <= 255)
    (x_min, x_max), (y_min, y_max) = min_max(binary)
    # src_w, src_h = x_max-x_min, y_max-y_min

    # if src_w / src_h == dest_w / dest_h:
    return fromimage(image)[x_min:x_max, y_min:y_max]
    #
    # if (src_w / src_h) > (dest_w / dest_h):  # need to pad height
    #     print("A")
    #     pad_y = (dest_h / dest_w) * src_w - src_h
    #     return image[(y_min-pad_y/2):(y_max+pad_y/2), x_min:x_max]
    # else:
    #     print("B")
    #     pad_x = (dest_w / dest_h) * src_h - src_w
    #     return image[(x_min-pad_x/2):(x_max-pad_x/2), y_min:y_max]


#
# def prep_image(image):
#     image = crop(image)
#     image = imresize(image, )


if __name__ == "__main__":
    from skimage import io
    from mkdata_random import get_files

    images = list(get_files("/home/hannes/Data/firmasSINTESISmanuscritas"))
    num_imgs = len(images)

    image = images[0]
    ratio = 200/120


    avg_x = 0
    avg_y = 0
    count = 0
    for f in images:
        with Image.open(f) as image:
            print("{}/{} - {}".format(count, num_imgs, f))#, end='\r')
            count += 1
            image = crop(image, (1.025,2))
            imsave("/home/hannes/tmp/images/{}.png".format(count), image)


    #     thresh = threshold_otsu(image)
    #     binary = np.asarray(image <= thresh)
    #     (x_min, x_max), (y_min,y_max) = min_max(binary)
    #     avg_x += (x_max - x_min)
    #     avg_y += (y_max - y_min)
    #     count += 1
    #
    # print()
    # print (avg_x, avg_y, avg_x / num_imgs, avg_y / num_imgs)
    # 86551127 168976381 400.69966203703706 782.2980601851851
