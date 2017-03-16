from cStringIO import StringIO
import numpy as np
import os
import subprocess
import glob
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
data = []
SNPS = 10

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

data = np.array(data).astype(int)

LABELS = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,
3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,
7,7,7,7,7,7,7,7])

def entropy(X):
    '''
    unique, counts = np.unique(X, return_counts=True)
    count = dict(zip(unique, counts))

    total = float(len(X))
    s = 0
    for c in count:
        s -= (count[c]/total)*math.log(count[c]/total,2)
    return s
    '''
    unique, counts = np.unique(X, return_counts=True)
    counts = counts/float(len(X)) # calculates the probabilities
    entropy=stats.entropy(counts)  # input probabilities to get the entropy
    return entropy

def getLeftRight(X,Y):
    if X.shape != Y.shape:
        raise Exception('Mismatch')
    left = []
    right = []
    total = float(len(X))

    for x,y in zip(X,Y):
        if y==1:
            left.append(x)
        else:
            right.append(x)
    eleft, eright = 0,0
    if len(left)>0:
        eleft = (len(left)/total)*entropy(left)
    if len(right)>0:
        eright = (len(right)/total)*entropy(right)
    return eleft+eright

def IG(X,Y):
    return entropy(X)-getLeftRight(X,Y)

gain = []
ENT = entropy(LABELS)
for snp in data:
    gain.append(ENT-getLeftRight(LABELS,snp))

np.save('gain.npy',gain)
#aggregatedIG = np.fromfile('informative/aggIG')

plt.scatter(range(len(gain)),gain, s=0.1)

plt.show()
