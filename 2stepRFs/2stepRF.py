import numpy as np
import pandas
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency
from sklearn.cross_validation import LeaveOneOut
from sklearn.model_selection import KFold

import logging
logging.basicConfig(filename='loo.log',level=logging.DEBUG)

SNPS = 160000
R = 30
mtry = 0.5
THETA = 0.05
THETA_CHI = 0.00000005
RE_RANK = True

LABELS = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,
3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,
7,7,7,7,7,7,7,7])
N_LABEL = len(np.unique(LABELS))



print 'Reading txt file...'
dataO = []
shadowO = []
with open('data/in.txt') as f:
    lines = f.readlines()
    i = 0
    for l in lines:
        if l[0] != '#' and i < SNPS:
            i+=1
            s = l.split()
            dataO.append(s[9:])
            shadowO.append(np.random.permutation(s[9:]))

dataO = np.array(dataO)

kf = KFold(n_splits=5)
Kf = 0
loo = kf.split(np.array(dataO).T)
#loo = LeaveOneOut(len(np.array(dataO).T))
#testSplits = loo.split(np.array(dataO).T)

for train, test in loo:
    Kf +=1
    if True:
        data = np.array(dataO)[:,train]
        shadow = np.array(shadowO)[:,train]
        trainLabels = LABELS[train]
        if RE_RANK:

            print 'Appending shadows...'
            data = np.concatenate((data,shadow))
            print 'Transposing...'
            data = np.array(data).transpose()
            print data.shape

            print 'Generating '+str(R)+' replicate forests...'
            RImportanceScores = []
            for x in range(R):
                clf = RandomForestClassifier(n_estimators=100, max_features=int(mtry*SNPS*2), n_jobs=4, verbose=True)
                clf.fit(data, trainLabels)
                RImportanceScores.append(clf.feature_importances_)
            RImportanceScores = np.array(RImportanceScores)

            print 'Calculating p-values with Wilcoxon test...'
            shadowImportanceScores = RImportanceScores[:,SNPS:]
            shadowImportanceMax = shadowImportanceScores.max(axis=1)
            print 'Shadow importances are: ', shadowImportanceMax
            pValues = []

            for SNPScores, shadowS in zip(RImportanceScores[:,:SNPS].T, RImportanceScores[:,SNPS:].T):
                if not np.array_equal(SNPScores, shadowS):
                    #_,p = mannwhitneyu(SNPScores, shadow, alternative='greater') #1425
                    _,p = wilcoxon(SNPScores, shadowS) #603
                    #_,p = wilcoxon(SNPScores, shadowImportanceMax)
                    #TODO: Wilcoxon test is two sided...........!
                    pValues.append(p)
                else:
                    pValues.append(float('inf'))
            import matplotlib.pyplot as plt
            plt.plot(pValues)
            plt.ylabel('some numbers')
            plt.show()

            print 'Ranking SNPs...'
            pValueIndexes = []
            for x,val in enumerate(pValues):
                if val <= THETA: #TODO: Weird
                    pValueIndexes.append(x)

            np.save('rankedSNP.txt', pValueIndexes)

            #WilPassedSNPs = data[:,pValueIndexes]
            #print WilPassedSNPs.shape

            #df = pandas.DataFrame(WilPassedSNPs)
            #df.to_csv('passed.csv', header=False, index=False)

        else:
            #WilPassedSNPs = pandas.read_csv('passed.csv', header=None)
            #WilPassedSNPs = WilPassedSNPs.as_matrix()
            #print WilPassedSNPs.shape
            pValueIndexes = np.load('rankedSNP.txt')


        SNPChiSquaredPassed = []
        SNPChiSquaredFailed = []
        for SNPidx in pValueIndexes:
            table = np.zeros(shape=(2, N_LABEL), dtype=int)
            for s,l in zip(dataO[SNPidx], LABELS):
                table[int(s),int(l)] += 1
            _,p,_,_ = chi2_contingency(table) #TODO: check parameters of this..
            if p <= THETA_CHI:
                SNPChiSquaredPassed.append(SNPidx)
            else:
                SNPChiSquaredFailed.append(SNPidx)

        snpIndexes = np.concatenate((SNPChiSquaredPassed, SNPChiSquaredPassed, SNPChiSquaredFailed))

        #snpIndexes = pValueIndexes

        reducedData = dataO[snpIndexes].T

        clf = RandomForestClassifier(n_estimators=100, verbose=True, n_jobs=-1)
        clf.fit(reducedData[train], trainLabels)
        pred = clf.predict(reducedData[test])
        res =  (pred == LABELS[test]).sum()/float(len(pred))
        logging.info(str(Kf))
        logging.info(reducedData.shape)
        logging.info(str(res))
        logging.info(str(LABELS[test]))
        logging.info(str(pred[0]))
