import csv
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
import openSplits

np.set_printoptions(edgeitems=15)

splits, LABELS, names, weights = openSplits.load()

for numEst in [60]:
    loo = LeaveOneOut()
    trainres = []
    res = []
    predRes = []
    for train, test in loo.split(splits):
        clf = RandomForestClassifier(n_estimators=numEst,
                                    verbose=False,
                                    n_jobs=-1)
        clf.fit(splits[train], LABELS[train])
        pred = np.array(clf.predict(splits[test]))
        predRes.append(np.array(clf.score(splits[train], LABELS[train])))
        res.append(pred[0])
    for i in range(len(LABELS)):
        print res[i], LABELS[i], names[i]
    print (np.array(res) == LABELS).sum()/float(len(LABELS))
