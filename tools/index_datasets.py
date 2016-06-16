import os
import sys
import pickle


def get_samples(data_dir):
    """returns a generator on lists of files per class in directory"""
    for d in os.listdir(data_dir):
        path = os.path.join(data_dir, d)
        if not os.path.isdir(path):
            continue
        files = os.listdir(path)
        yield (d, [os.path.join(path, f) for f in files
                                         if '.png' in f.lower()])


def get_dict(data_dir):
    """returns a dictionary with samples found in data_dir
       expects data_dir to contain one subfolder per class"""
    return {int(f[0]): f[1] for f in get_samples(data_dir)}


if __name__ == '__main__':
    if len(sys.argv) < 2 or '-h' in sys.argv or '--help' in sys.argv:
        print("Usage:\t{} (data_dir)+".format(sys.argv[0]))
        exit()
    data_paths = [sys.argv[i] for i in range(1, len(sys.argv))]
    out = "index.pkl"

    genuine, forged = [], []
    for path in data_paths:
        path_genuine = os.path.join(path, 'Genuine')
        path_forged = os.path.join(path, 'Forged')
        genuine.append(get_dict(path_genuine))
        forged.append(get_dict(path_forged))

    genuine_flat = [s for g in genuine for s in g.values()]
    forged_flat = [s for g in forged for s in g.values()]

    samples = {
        'Genuine': {label: genuine_flat[label] for label in range(len(genuine_flat))},
        'Forged': {label: forged_flat[label] for label in range(len(forged_flat))},
    }
    
    pickle.dump(samples, open(out, 'wb'))
