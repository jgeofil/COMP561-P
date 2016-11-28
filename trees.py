import numpy as np
import pandas
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency
from sklearn.model_selection import LeaveOneOut

SNPS = 800000
R = 30
mtry = 0.5
THETA = 0.05
THETA_CHI = 0.00000005
RE_RANK = False

LABELS = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,
3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,
7,7,7,7,7,7,7,7])
N_LABEL = len(np.unique(LABELS))

if RE_RANK:
    print 'Reading txt file...'
    data = []
    shadow = []
    with open('data/in.txt') as f:
        lines = f.readlines()
        i = 0
        for l in lines:
            if l[0] != '#' and i < SNPS:
                i+=1
                s = l.split()
                data.append(s[9:])
                shadow.append(np.random.permutation(s[9:]))
    print 'Appending shadows...'
    data = data + shadow
    print 'Transposing...'
    data = np.array(data).transpose()
    print data.shape

    print 'Generating '+str(R)+' replicate forests...'
    RImportanceScores = []
    for x in range(R):
        clf = RandomForestClassifier(n_estimators=100, max_features=int(mtry*SNPS*2), n_jobs=-1, verbose=True)
        clf.fit(data, LABELS)
        RImportanceScores.append(clf.feature_importances_)
    RImportanceScores = np.array(RImportanceScores)

    print 'Calculating p-values with Wilcoxon test...'
    shadowImportanceScores = RImportanceScores[:,SNPS:]
    shadowImportanceMax = shadowImportanceScores.max(axis=1)
    print 'Shadow importances are: ', shadowImportanceMax
    pValues = []

    print RImportanceScores

    for SNPScores, shadow in zip(RImportanceScores[:,:SNPS].T, RImportanceScores[:,SNPS:].T):
        if not np.array_equal(SNPScores, shadow):
            #_,p = mannwhitneyu(SNPScores, shadow, alternative='greater') #1425
            _,p = wilcoxon(SNPScores, shadow) #603
            #_,p = wilcoxon(SNPScores, shadowImportanceMax)
            #TODO: Wilcoxon test is two sided...........!
            pValues.append(p)
        else:
            pValues.append(float('inf'))

    print 'Ranking SNPs...'
    pValueIndexes = []
    print pValues
    for x,val in enumerate(pValues):
        if val <= THETA: #TODO: Weird
            pValueIndexes.append(x)

    WilPassedSNPs = data[:,pValueIndexes]
    print WilPassedSNPs.shape

    df = pandas.DataFrame(WilPassedSNPs)
    df.to_csv('passed.csv', header=False, index=False)

else:
    WilPassedSNPs = pandas.read_csv('passed.csv', header=None)
    WilPassedSNPs = WilPassedSNPs.as_matrix()
    print WilPassedSNPs.shape

SNPChiSquared = []
for SNP in WilPassedSNPs.T:
    table = np.zeros(shape=(2, N_LABEL), dtype=int)
    for s,l in zip(SNP, LABELS):
        table[int(s),int(l)] += 1
    _,p,_,_ = chi2_contingency(table) #TODO: check parameters of this..
    SNPChiSquared.append(p)


print 'Ranking SNPs...'
pValueIndexes = []
notpValueIndexes = []
for x,val in enumerate(SNPChiSquared):
    if val <= THETA_CHI:
        pValueIndexes.append(x)
    else:
        notpValueIndexes.append(x)

GreatSNPs = WilPassedSNPs[:,pValueIndexes]
OkSNPs = WilPassedSNPs[:,notpValueIndexes]
print OkSNPs.shape
print GreatSNPs.shape


loo = LeaveOneOut()
res = []
proba = []
for train, test in loo.split(np.concatenate((OkSNPs,GreatSNPs,GreatSNPs), axis=1)):
    clf = RandomForestClassifier(n_estimators=100, verbose=True, n_jobs=-1)
    clf.fit(WilPassedSNPs[train], LABELS[train])
    res.append(clf.predict(WilPassedSNPs[test])[0])
    proba.append(clf.predict_proba(WilPassedSNPs[test])[0])
for i in range(len(LABELS)-1):
    print res[i], LABELS[i], max(proba[i])

print len(WilPassedSNPs)
print (res == LABELS).sum()/float(len(WilPassedSNPs))
