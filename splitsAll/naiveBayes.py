import csv
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import MultinomialNB
import openSplits

np.set_printoptions(edgeitems=15)

splits, LABELS, names, weights = openSplits.load()

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
