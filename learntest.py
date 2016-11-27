import csv
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

LABELS = [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,
3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,
7,7,7,7,7,7,7,7]

names = []
weights = []
splits = []

with open('split.txt', 'rb') as csvfile:
     lines = csv.reader(csvfile, delimiter='\t')
     for i, row in enumerate(lines):
        if i > 1 and len(row):
            splits.append(row[2:])
            weights.append(row[1])
        elif i == 1:
            names = row [2:]

weights= np.array(weights).astype('float')

splits = np.array(splits).transpose().astype('int')

splits = np.array(splits)
LABELS = np.array(LABELS)

'''
loo = LeaveOneOut()
res = []
proba = []
for train, test in loo.split(splits):
    clf = RandomForestClassifier(n_estimators=50, min_samples_split=2, verbose=False, max_features=None)
    clf.fit(splits[train], LABELS[train])
    res.append(clf.predict(splits[test])[0])
    proba.append(clf.predict_proba(splits[test])[0])
    #print names[test], res[test], LABELS[test], proba[test]
for i in range(len(LABELS)):
    print names[i], res[i], LABELS[i], max(proba[i])

print (np.array(res) == LABELS).sum()/float(len(splits))
'''

'''
loo = LeaveOneOut()
res = []
for train, test in loo.split(splits):
    clf = MLPClassifier(solver='sgd', alpha=1e-4, hidden_layer_sizes=(200, 10), random_state=1, verbose=False, max_iter=400)
    clf.fit(splits[train], LABELS[train])
    res.append(clf.predict(splits[test])[0])
    print names[test], res[test], LABELS[test]
for i in range(len(LABELS)):
    print names[i], res[i], LABELS[i]

print (np.array(res) == LABELS).sum()/float(len(splits))
'''

'''
def funcDist(X,Y):
    dist = sum([0 if x == y else w for (x,y,w) in zip(X,Y,weights)])
    return dist


n_neighbors = 4
loo = LeaveOneOut()

for dist in ['jaccard','matching','dice','kulsinski','rogerstanimoto','russellrao','sokalmichener','sokalsneath']:
    res = []
    for train, test in loo.split(splits):
        clf = neighbors.KNeighborsClassifier(n_neighbors, metric=dist, weights='distance')
        clf.fit(splits[train], LABELS[train])
        res.append(clf.predict(splits[test])[0])
    #for i in range(len(LABELS)):
    #    print names[i], res[i], LABELS[i]
    print (np.array(res) == LABELS).sum()/float(len(splits))

'''


def funcDist(X,Y):
    dist = sum([0 if x == y else w for (x,y,w) in zip(X,Y,weights)])
    return dist

n_neighbors = 4
loo = LeaveOneOut()

for dist in [1]:
    res = []
    for train, test in loo.split(splits):
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance', metric=funcDist)
        clf.fit(splits[train], LABELS[train])
        res.append(clf.predict(splits[test])[0])
    #for i in range(len(LABELS)):
    #    print names[i], res[i], LABELS[i]
    print (np.array(res) == LABELS).sum()/float(len(splits))




'''
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
pca.fit(splits)
comp = pca.components_
plt.scatter(comp[0], comp[1])
plt.show()
'''
