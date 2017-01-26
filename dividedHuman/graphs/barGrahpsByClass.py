import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import Counter

resMat = np.load('resmat.npy')
print resMat.shape

LABELS = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,
3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,
7,7,7,7,7,7,7,7])
print Counter(LABELS)
contigs = [1,2,3,4,5,6,7,8,9,10,11,12,13,33]
mainPlot = []

for i, chrom in enumerate(resMat):
    plotList = []
    chrom = chrom.T
    for k, cat in enumerate(chrom):
        outPos = []
        outNeg = []

        for j, ind in enumerate(cat):
            if k == LABELS[j]:
                outPos.append(ind)
            else:
                outNeg.append(ind)
        plotList.append(outPos)
        #plotList.append(outNeg)
    mainPlot.append(plotList)


fig, axes = plt.subplots(nrows=7, ncols=2)
fig.text(0.5, 0.04, 'Origin classes', ha='center')
fig.text(0.04, 0.5, 'Predicted probabilities', va='center', rotation='vertical')
for i in range(7):
    for j in range(2):
        axes[i,j].set_title(contigs[i+(j*7)])
        axes[i,j].yaxis.set_ticks(np.arange(0.25,1,0.25))
        plot = axes[i,j].boxplot(mainPlot[i+(j*7)], patch_artist=True)
        plt.setp(plot['boxes'], color='black')
        plt.setp(plot['medians'], color='black')
        plt.setp(plot['whiskers'], color='black')
        axes[i,j].grid(axis='y')
        axes[i,j].axis([0,9, 0, 1])


        if j == 1:
            axes[i,j].set_yticklabels([])

        axes[i,j].set_xticklabels(['I', 'Q', 'O', 'E', 'M', 'S', 'A', 'B'])
        if not i == 6:
            axes[i,j].set_xticklabels([])
        #axes[i,j].twinx().semilogy(range(1,15),[249007,202653,153526,141551,113360,120764,111483,114486,100544,95923,85995,69275,51206,183])
        for k, patch in enumerate(plot['boxes']):
            patch.set_facecolor(cm.get_cmap('Set1')(k/8.,1))

fig.subplots_adjust(hspace=0.25)
fig.subplots_adjust(wspace=0.05)
fig.subplots_adjust(left=0.115)
fig.set_size_inches(8,10.5)
fig.savefig('classes.eps', dpi=500)
