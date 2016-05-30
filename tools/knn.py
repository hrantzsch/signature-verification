from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import pickle
import sys
import numpy as np

# TODO sklearn has a KNN-Classifier that I could use rather than implementing
# my own...


def load_data(path):
    data = pickle.load(open(path, 'rb'))
    keys = sorted(list(data.keys()))
    for k in keys:
        for s in data[k]:
            yield((int(k), s))


def vote(voices):
    """Accumulate predictions for a single test sample's nn predictions;
       Not needed anymore when I use KNeighborsClassifier directly."""
    #    usage example:
    #    majority = list(map(vote, map(lambda x: y[x], nbs)))
    return np.argmax(np.bincount(voices))


if __name__ == '__main__':
    print("Loading data...")
    data = list(load_data(sys.argv[1]))
    np.random.shuffle(data)
    n_train = int(len(data) * 0.9)
    y, X = zip(*data[:n_train])  # unzip tuples for training
    yt, Xt = zip(*data[n_train:])

    # === train ===
    print("Training...")
    nn = KNeighborsClassifier(n_neighbors=5)
    nn.fit(X, y)

    pickle.dump(nn, open('knn.pkl', 'wb'))

    # === test ===
    print("Testing...")
    nbs = nn.predict(Xt)
    # correct = majority == yt
    # print("\n========")
    # print("Accuracy: {}".format(sum(correct) / len(correct)))

    correct = 0
    predictions = np.array(list(map(lambda x: y[x], nbs)))
    for val in np.unique(yt):
        this_class = predictions[yt == val].flatten()
        print("{}: {}".format(val, vote(this_class)))
        if val == vote(this_class):
            correct += 1
    print("Accuracy: {}".format(correct / len(np.unique(yt))))

    import pdb; pdb.set_trace()
