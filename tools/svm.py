from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import pickle
import sys
import numpy as np


def load_data(path):
    data = pickle.load(open(path, 'rb'))
    keys = sorted(list(data.keys()))
    X = []
    y = []
    for k in keys:
        for x in data[k]:
            X.append(x)
            y.append(k[-3:])  # HACK
            # y.append(k)
    return np.array(X, np.float32), np.array(y, np.int32)

if __name__ == '__main__':
    X, y = load_data(sys.argv[1])
    Xt, yt = load_data(sys.argv[2])

    # === train ===
    clf = OneVsOneClassifier(svm.LinearSVC(verbose=1,
                                           max_iter=10000,
                                           dual=False,
                                           ), 5)
    clf.fit(X, y)
    pickle.dump(clf, open('svm.pkl', 'wb'))

    # clf = pickle.load(open('svm.pkl', 'rb'))

    # === test ===
    prediction = clf.predict(Xt)
    correct = prediction == yt
    print("\n========")
    print("Accuracy: {}".format(sum(correct) / len(correct)))
