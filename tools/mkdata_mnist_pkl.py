"""This script creates a pickle file of two lists following the form
[
    [('0', '<path>'), ('0', '<path>'), ...],
    [('1', '<path>'), ...],
    ...
]
They are used for indexing mnist samples by mnist_loader.py.
"""

import pickle
import os
import sys


def ensure_not_exists(path):
    if os.path.exists(path):
        print("Error: {} exists!".format(path))
        raise FileExistsError


def create_list(sample_set):
    out = []
    for entry in sorted(os.listdir(sample_set)):
        sub_dir = os.path.join(sample_set, entry)
        if not os.path.isdir(sub_dir):
            continue

        sub_list = []
        for f in os.listdir(sub_dir):
            if '.png' in f:
                sub_list.append((entry, os.path.join(sub_dir, f)))
        out.append(sub_list)
    return out


for sub_set in ['train', 'test']:
    full_path = os.path.join(sys.argv[1], sub_set)
    out_path = sub_set + '.pkl'
    ensure_not_exists(out_path)
    out_list = create_list(full_path)
    pickle.dump(out_list, open(out_path, 'wb'))
