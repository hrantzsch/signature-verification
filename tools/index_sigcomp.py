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
        yield (d, [os.path.join(path, f) for f in files if '.png' in f])


if __name__ == '__main__':
    data = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else 'sigcomp_index.pkl'
    path_genuine = os.path.join(data, 'Genuine')
    path_forged = os.path.join(data, 'Forged')

    genuine, forgeries = {}, {}
    for f in get_samples(path_genuine):
        genuine[int(f[0])] = f[1]
    for f in get_samples(path_forged):
        forgeries[int(f[0])] = f[1]

    samples = {'Genuine': genuine, 'Forged': forgeries}
    pickle.dump(samples, open(out, 'wb'))
