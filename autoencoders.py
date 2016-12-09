
from cStringIO import StringIO
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

data = []

with open('data/in.txt') as f:
    lines = f.readlines()
    for l in lines:
        if l[0] != '#':
            s = l.split()
            data.append(s[9:])

LABELS = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,
3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,
7,7,7,7,7,7,7,7])

data = np.array(data).transpose()

loo = LeaveOneOut()
res = []
for train, test in loo.split(data):
    clf = RandomForestClassifier(n_estimators=100, min_samples_split=2, verbose=True)
    clf.fit(data[train], LABELS[train])
    res.append(clf.predict(data[test])[0])
    print res[test], LABELS[test]
for i in range(len(LABELS)):
    print res[i], LABELS[i]

print (np.array(res) == LABELS).sum()/float(len(data))
