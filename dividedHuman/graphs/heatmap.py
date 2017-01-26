import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import Counter

resMat = np.load('resmat.npy')
print resMat.shape

np.set_printoptions(precision=2)

LABELS = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,
3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,
7,7,7,7,7,7,7,7])

print Counter(LABELS)

mainPlot = np.zeros((8,8))

sums = np.zeros(8)
for i, chrom in enumerate(resMat):
    for j, ind in enumerate(chrom):
        for k, cat in enumerate(ind):
            mainPlot[k, LABELS[j]] += cat
            sums[LABELS[j]] +=1


print 14*83

mainPlot = mainPlot/(np.array([7,19,7,7,10,10,8,15])*14)

print mainPlot[:, 5].sum()
print mainPlot.sum(axis=0)
print mainPlot.sum(axis=1)
fig, ax = plt.subplots()
heatmap = ax.pcolor(mainPlot, cmap='viridis')

ax.set_xticks(np.arange(8) + 0.5, minor=False)
ax.set_yticks(np.arange(8) + 0.5, minor=False)
ax.invert_yaxis()

ax.set_xlabel('True class')
ax.set_ylabel('Predicted class')

ax.set_xticklabels(['I', 'Q', 'O', 'E', 'M', 'S', 'A', 'B'], minor=False)
ax.set_yticklabels(['I', 'Q', 'O', 'E', 'M', 'S', 'A', 'B'], minor=False)

#legend
cbar = plt.colorbar(heatmap)
cbar.ax.axis([0,1,0,1])
#cbar.ax.set_yticklabels(['0','1','2','>3'])
fig.subplots_adjust(left=0.20)
fig.subplots_adjust(bottom=0.20)
fig.set_size_inches(3,3)
fig.savefig('heat.eps', dpi=500)
