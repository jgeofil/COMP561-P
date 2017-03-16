
from cStringIO import StringIO
import numpy as np
import os
import subprocess
import glob
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
data = []
SNPS = 100000

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

splits = []
with open('informative/split.txt') as f:
    lines = f.readlines()
    i = 0
    for l in lines:
        i+=1
        if i > 2:
            s = l.split()
            splits.append(s[2:])
splits = np.array(splits).astype(int)
print splits.shape

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

def getBaseEntropies(X):
    ent = []
    for row in X:
        ent.append(entropy(row))
    return ent

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

baseE = getBaseEntropies(splits)
aggregatedIG = []
for snp in data:
    aggr = 0
    for E,split in zip(baseE,splits):
        aggr += E-getLeftRight(split,snp)
    aggregatedIG.append(aggr)
aggregatedIG = np.array(aggregatedIG)

#np.tofile(aggregatedIG, 'informative/aggIG')
#aggregatedIG = np.fromfile('informative/aggIG')

plt.scatter(range(len(aggregatedIG)),aggregatedIG, s=0.1)

plt.show()
