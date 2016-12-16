import numpy as np
from sklearn import neighbors
from sklearn.model_selection import LeaveOneOut
import math
import openSplits

np.set_printoptions(edgeitems=15)

splits, LABELS, names, weights = openSplits.load()


def funcDist(X,Y):
    return sum([ 0 if x==y else 1 for x,y,e in zip(X,Y,weights)])

loo = LeaveOneOut()
for n_neighbors in [4]:
    res = []
    for train, test in loo.split(splits):
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance', metric=funcDist)
        clf.fit(splits[train], LABELS[train])
        res.append(clf.predict(splits[test])[0])
    for i in range(len(LABELS)):
        print res[i], LABELS[i], names[i]
    print (np.array(res) == LABELS).sum()/float(len(LABELS))
    print LABELS[np.array(res) != LABELS]
