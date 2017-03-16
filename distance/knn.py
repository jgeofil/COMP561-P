import csv
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

LABELS = [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,
3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,
7,7,7,7,7,7,7,7]

dist = []

with open('distance/dist.txt', 'rb') as csvfile:
     lines = csv.reader(csvfile, delimiter='\t')
     for i, row in enumerate(lines):
        if len(row)>2:
            dist.append(row)

dist = np.array(dist).transpose().astype('float')
LABELS = np.array(LABELS)

def funcDist(X,Y):
    x0 = np.argmin(X)
    y0 = np.argmin(Y)

    dist = abs(X[y0])

    return dist

n_neighbors = 1
loo = LeaveOneOut()

res = []
for train, test in loo.split(dist):
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform', metric=funcDist)
    clf.fit(dist[train], LABELS[train])
    res.append(clf.predict(dist[test])[0])
for i in range(len(LABELS)):
    print res[i], LABELS[i]
print (np.array(res) == LABELS).sum()/float(len(dist))
