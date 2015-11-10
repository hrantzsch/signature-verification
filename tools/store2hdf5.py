import h5py
import numpy as np
import csv
from PIL import Image


def store2hdf5(h5file, data, labels, chunksize):

    try:
        # get the datasets
        data_dataset = h5file["data"]
        label_dataset = h5file['label']
        # set the start indices
        start_data = data_dataset.shape[0]
        start_label = label_dataset.shape[0]
        # resize the datasets so that the new data can fit in
        data_dataset.resize(start_data + data.shape[0], 0)
        label_dataset.resize(start_data + labels.shape[0], 0)
    except KeyError:
        # create new datasets in hdf5 file
        data_shape = data.shape
        data_dataset = h5file.create_dataset(
            "/data",
            shape=data_shape,
            maxshape=(
                None,
                data_shape[1],
                data_shape[2],
                data_shape[3],
            ),
            dtype="f",
            chunks=(chunksize,1,100,200),
        )
        label_shape = labels.shape
        label_dataset = h5file.create_dataset(
            "/label",
            shape=label_shape,
            maxshape=(
                None,
            ),
            dtype="f",
            chunks=(chunksize,),
        )
        # set the start indices in fourth dimension
        start_data = 0
        start_label = 0

    if label_dataset is not None and data_dataset is not None:
        # write the given data into the hdf5 file
        data_dataset[start_data:start_data + data.shape[0], :, :, :] = data
        label_dataset[start_label:start_label + labels.shape[0]] = labels


def image_to_np_array(image):
    pixels = np.array(image, dtype=np.float32)
    return normalize_image(pixels[np.newaxis, ...])[np.newaxis, ...]


def normalize_image(np_image):
    # normalize the image to contain values from 0 to 1 in each channel
    minval = np_image.min()
    maxval = np_image.max()
    if minval != maxval:
        np_image -= minval
        np_image *= (1.0 / (maxval-minval))
    return np_image
