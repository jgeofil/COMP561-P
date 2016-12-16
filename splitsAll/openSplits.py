import numpy as np
import csv

def load():

    LABELS = [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,
    3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7]

    names = []
    weights = []
    splits = []

    with open('splitsAll/split.txt', 'rb') as csvfile:
         lines = csv.reader(csvfile, delimiter='\t')
         for i, row in enumerate(lines):
            if i > 1 and len(row):
                splits.append(row[2:])
                weights.append(row[1])
            elif i == 1:
                names = row [2:]

    weights= np.array(weights).astype('float')

    splits = np.array(splits).transpose().astype('int')

    splits = np.array(splits)
    LABELS = np.array(LABELS)

    return splits, LABELS, names, weights
