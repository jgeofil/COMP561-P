import csv
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
import os
import glob
import csv

cwd = os.getcwd()

files = glob.glob('data/divided/splits*.nex')

LABELS = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,
3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,
7,7,7,7,7,7,7,7])

dataList = []

if True:

    for iname in range(len(files)):
        fname = 'data/divided/splits'+str(iname)+'.nex'

        splitsList = []

        with open(fname, 'rb') as f:
             lines = f.readlines()
             splits = False
             number = 0
             matrix = False
             done = False
             splitsMatrix = []
             for i, row in enumerate(lines):
                if splits and matrix and not done:
                    if len(row) <= 5:
                        done = True
                    else:
                        row = np.array(row.split()[3:])
                        row[-1] = row[-1].replace(',','')
                        row = row.astype(int)-1
                        binary = np.array([0]*number)
                        binary[row] =1
                        splitsList.append(binary)

                elif row[:13] == 'BEGIN Splits;':
                    splits = True
                elif splits and not number:
                    digits = row.split()[1].split('=')[1]
                    number = int(digits)
                elif number and row[:6] == 'MATRIX':
                    matrix = True

        splitsList = np.array(splitsList).transpose()
        dataList.append(splitsList)
        print splitsList.shape

    resMat = []
    for data in dataList:
        loo = LeaveOneOut()
        res = []
        for train, test in loo.split(data):
            clf = RandomForestClassifier(n_estimators=50,
                                        verbose=False,
                                        n_jobs=-1)
            clf.fit(data[train], LABELS[train])
            print np.argmax(clf.predict_proba(data[test])[0]) == LABELS[test]
            res.append(clf.predict_proba(data[test])[0].tolist())
        resMat.append(res)

    np.save('resmat',resMat)
resMat = np.load('resmat.npy')
print resMat.shape

flat = []
for cat, catvals in enumerate(resMat):
    for ind, indvals in enumerate(catvals):

        flat.append([cat, ind] + indvals.tolist())

print flat

f = open('fullprob.csv', 'w')

spamwriter = csv.writer(f, delimiter=',')
for j, seq in enumerate(flat):
    spamwriter.writerow(seq)


'''
count = resMat.shape[0]
resMat = np.sum(resMat, axis=0)


resMat = resMat/float(count)
#print resMat

argMax = [np.argmax(r) for r in resMat]
print argMax == LABELS


f = open('prob.tsv', 'w')

spamwriter = csv.writer(f, delimiter=' ')
for j, seq in enumerate(resMat):
    spamwriter.writerow(seq)
'''
