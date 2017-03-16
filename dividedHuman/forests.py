import csv
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
import os
import glob
import csv

cwd = os.getcwd()

files = glob.glob('data/human/splits.chr*.nex')

label = open('data/human/outsuper.txt')

labels =[]
for l in label.readlines():
    labels.append(l.rstrip())

print labels
print len(labels)
LABELS = np.array(labels)

dataList = []
classes = []

if True:

    for iname in files:
        fname = iname
        print fname
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
        trueclass = []
        loo = LeaveOneOut()
        res = []
        for train, test in loo.split(data):
            clf = RandomForestClassifier(n_estimators=60,
                                        verbose=False,
                                        n_jobs=-1)
            clf.fit(data[train], LABELS[train])
            classes = clf.classes_
            np.save('classes', classes)
            #print clf.predict(data[test])[0],  LABELS[test][0]
            trueclass.append(clf.predict(data[test])[0])
            res.append(clf.predict_proba(data[test])[0].tolist())
        resMat.append(res)
        print trueclass
        print np.array(trueclass == LABELS).sum()/float(156)

    np.save('resmat',resMat)


resMat = np.load('resmat.npy')
classes = np.load('classes.npy')

avgMat = np.average(resMat, axis=0)
#for oneEst in resMat:
argClass = map(np.argmax, avgMat)
argName = map(lambda x: classes[x], argClass)
print argName
print (argName == LABELS).sum()/float(156)
