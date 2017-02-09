import csv
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
import os
import glob
import csv

cwd = os.getcwd()

files = glob.glob('data/human/splits1.nex')

label = open('data/human/outpops.txt')

labels =[]
for l in label.readlines():
    labels.append(l.rstrip())

print labels
LABELS = np.array(labels)

dataList = []

if True:

    for iname in range(len(files)):
        fname = 'data/human/splits'+str(iname)+'.nex'

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
    trueclass = []
    for data in dataList:
        loo = LeaveOneOut()
        res = []
        for train, test in loo.split(data):
            clf = RandomForestClassifier(n_estimators=10,
                                        verbose=False,
                                        n_jobs=-1)
            clf.fit(data[train], LABELS[train])
            print clf.predict(data[test])[0],  LABELS[test][0]
            trueclass.append(clf.predict(data[test])[0] == LABELS[test][0])
            res.append(clf.predict_proba(data[test])[0].tolist())
        resMat.append(res)

    np.save('resmat',resMat)
    print np.array(trueclass).sum()/float(156)

resMat = np.load('resmat.npy')
print resMat.shape

flat = []
for cat, catvals in enumerate(resMat):
    for ind, indvals in enumerate(catvals):

        flat.append([cat, ind] + indvals.tolist())

print flat

f = open('fullprob.csv', 'w')
'''
spamwriter = csv.writer(f, delimiter=',')
for j, seq in enumerate(flat):
    spamwriter.writerow(seq)

catsnum = []
for i in range(26):
    for j in range(5):
        catsnum.append(i)
catsnum = np.array(catsnum)


'''
count = resMat.shape[0]
resMat = np.sum(resMat, axis=0)


resMat = resMat/float(count)
#print resMat

argMax = [np.argmax(r) for r in resMat]
print argMax

classes = []
for i in range(26):
    for j in range(6):
        classes.append(i)
classes = np.array(classes)
print (classes == argMax).sum()


f = open('prob.tsv', 'w')

spamwriter = csv.writer(f, delimiter=' ')
for j, seq in enumerate(resMat):
    spamwriter.writerow(seq)
