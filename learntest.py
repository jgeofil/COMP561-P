import csv
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from collections import Counter
import math
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB

np.set_printoptions(edgeitems=15)

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

#print splits

splits = np.array(splits)
LABELS = np.array(LABELS)
'''
loo = LeaveOneOut()
res = []
for train, test in loo.split(splits):

    clf = AdaBoostClassifier(n_estimators=200)
    clf.fit(splits[train], LABELS[train])
    pred = np.array(clf.predict(splits[test]))
    res.append((pred == LABELS[test]).sum()/float(len(LABELS[test])))
print sum(res)/float(len(LABELS))
'''


def entro(cats):
    counts = Counter(cats)
    tot = []
    for c in np.unique(LABELS):
        ratio = counts[c]/float(len(cats))
        tot.append(ratio*math.log(ratio,2) if ratio else 0)
    return -sum(tot)


entropy = []
for s in splits.T:
    isone = s == 1
    a,b = LABELS[isone],LABELS[np.invert(isone)]
    #entropy.append(max([len(a)/float(len(b)), len(b)/float(len(a))]))
    entropy.append((len(a)/float(len(LABELS))*entro(a))+(len(b)/float(len(LABELS))*entro(b)))



'''


for numEst in range(10, 110, 10):
    loo = LeaveOneOut()
    trainres = []
    res = []
    predRes = []
    for train, test in loo.split(splits):
        clf = RandomForestClassifier(n_estimators=numEst,
                                    min_samples_split=2,
                                    verbose=False,
                                    max_features=None,
                                    n_jobs=-1)
        clf.fit(splits[train], LABELS[train])
        pred = np.array(clf.predict(splits[test]))
        predRes.append(np.array(clf.score(splits[train], LABELS[train])))
        res.append((pred == LABELS[test]).sum()/float(len(LABELS[test])))
    for i in range(len(LABELS)):
        print res[i], LABELS[i]
    print sum(res)/float(len(LABELS)), numEst, sum(predRes)/float(len(LABELS))



'''


def funcDist(X,Y):
    return sum([ 0 if x==y else e for x,y,e in zip(X,Y,weights)])

'''
loo = LeaveOneOut()
for n_neighbors in range(1,15):
    res = []
    for train, test in loo.split(splits):
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
        clf.fit(splits[train], LABELS[train])
        res.append(clf.predict(splits[test])[0])
    #for i in range(len(LABELS)):
    #    print res[i], LABELS[i], names[i]
    print (np.array(res) == LABELS).sum()/float(len(LABELS))
    print LABELS[np.array(res) != LABELS]
'''

loo = LeaveOneOut()
res = []
for train, test in loo.split(splits):
    clf =  MultinomialNB()
    clf.fit(splits[train], LABELS[train])
    res.append(clf.predict(splits[test])[0])
for i in range(len(LABELS)):
    print res[i], LABELS[i], names[i]
print ((np.array(res) == LABELS).sum())/float(len(LABELS))
print LABELS[np.array(res) != LABELS]
