
from cStringIO import StringIO
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

data = []
SNPS = 1000000

print 'Reading txt file...'

with open('data/in.txt') as f:
    lines = f.readlines()
    i = 0
    for l in lines:
        if l[0] != '#' and i < SNPS:
            i+=1
            s = l.split()
            data.append(s[9:])

print 'Opened'
LABELS = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,
3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,
7,7,7,7,7,7,7,7])

data = np.array(data).transpose()

print data.shape

featImp = []
for _ in range(5):
    clf = RandomForestClassifier(n_estimators=60,
                                verbose=True,
                                n_jobs=-1)
    clf.fit(data, LABELS)
    featImp.append(clf.feature_importances_)
featImp = sum(featImp)
print featImp.shape

import matplotlib.pyplot as plt
plt.scatter(range(len(featImp)),featImp, s=0.1)
plt.ylabel('some numbers')
plt.show()
