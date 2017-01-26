import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

resMat = np.load('resmat.npy')
print resMat.shape

LABELS = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,
3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,
7,7,7,7,7,7,7,7])

mainPlot = []
contigs = ['I', 'Q', 'O', 'E', 'M', 'S', 'A', 'B']

resMat = resMat.T
print resMat.shape

for k, cat in enumerate(resMat):
    cat = cat.T
    plotList = []
    for i, chrom in enumerate(cat):
        outPos = []
        outNeg = []

        for j, ind in enumerate(chrom):
            if k == LABELS[j]:
                outPos.append(ind)
            else:
                outNeg.append(ind)
        plotList.append(outPos)
        #plotList.append(outNeg)
    mainPlot.append(plotList)


fig, axes = plt.subplots(nrows=4, ncols=2)
fig.text(0.5, 0.04, 'Scaffolds', ha='center')
fig.text(0.04, 0.5, 'Predicted probabilities', va='center', rotation='vertical')
fig.text(0.95, 0.5, 'Quantity of SNPs', va='center', rotation='vertical', color='b')
for i in range(4):
    for j in range(2):
        axes[i,j].yaxis.set_ticks(np.arange(0.25,1,0.25))
        axes[i,j].set_title(contigs[i+(j*4)])
        plot = axes[i,j].boxplot(mainPlot[i+(j*4)], patch_artist=True)
        plt.setp(plot['boxes'], color='black')
        plt.setp(plot['medians'], color='black')
        plt.setp(plot['whiskers'], color='black')
        axes[i,j].grid(axis='y')
        if not i == 3:
            axes[i,j].set_xticklabels([])
        if j == 1:
            axes[i,j].set_yticklabels([])

        twin = axes[i,j].twinx()
        twin.semilogy(range(1,15),[249007,202653,153526,141551,113360,120764,111483,114486,100544,95923,85995,69275,51206,183])
        twin.set_yscale('log', color='b')
        twin.set_yticks([1000, 10000, 100000])
        if j == 0:
            twin.set_yticklabels([])

        axes[i,j].axis([0,15, 0, 1])

        for k, patch in enumerate(plot['boxes']):
            patch.set_facecolor(cm.get_cmap('gist_earth')(k/14.,0.8))

fig.subplots_adjust(hspace=0.2)
fig.subplots_adjust(wspace=0.05)
fig.subplots_adjust(left=0.115)
fig.set_size_inches(8,10.5)
fig.savefig('chroms.eps', dpi=500)
