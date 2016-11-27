import csv
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

LABELS = [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,
3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,
7,7,7,7,7,7,7,7]

dist = []

with open('dist.txt', 'rb') as csvfile:
     lines = csv.reader(csvfile, delimiter='\t')
     for i, row in enumerate(lines):
        if len(row)>2:
            dist.append(row)

dist = np.array(dist).transpose().astype('float')
LABELS = np.array(LABELS)

print dist
print  len(dist)


def funcDist(X,Y):
    x0 = np.argmin(X)
    y0 = np.argmin(Y)

    dist = sum([abs((x-y)) for (x,y) in zip(X,Y)])

    return dist

'''
n_neighbors = 3
loo = LeaveOneOut()

for metric in ['euclidian']:
    res = []
    for train, test in loo.split(dist):
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform', metric=funcDist)
        clf.fit(dist[train], LABELS[train])
        res.append(clf.predict(dist[test])[0])
    for i in range(len(LABELS)):
        print res[i], LABELS[i]
    print (np.array(res) == LABELS).sum()/float(len(dist))
'''

loo = LeaveOneOut()
res = []
for train, test in loo.split(dist):
    clf = MLPClassifier(solver='sgd', alpha=1e-4, hidden_layer_sizes=(200, 100), random_state=1, verbose=False, max_iter=200)
    clf.fit(dist[train], LABELS[train])
    res.append(clf.predict(dist[test])[0])
    print res[test], LABELS[test]
for i in range(len(LABELS)):
    print res[i], LABELS[i]

print (np.array(res) == LABELS).sum()/float(len(dist))
