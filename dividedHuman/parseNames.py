import numpy as np
import os
cwd = os.getcwd()

names = []
pops = []
superpops = []
with open('dividedHuman/samples.panel') as f:
    lines = f.readlines()
    count = 0
    for line in lines:
        if count > 0:
            sp = line.split()
            names.append(sp[0])
            pops.append(sp[1])
            superpops.append(sp[2])
        count += 1

poped = {}
keptn = []
keptpop = []
keptspop = []
keys = np.unique(pops)
for k in keys:
    poped[k] = 0

for n,p,sp in zip(names, pops, superpops):
    if poped[p] <= 5:
        poped[p] += 1
        keptn.append(n)
        keptpop.append(p)
        keptspop.append(sp)


out = open(cwd + '/data/human/outsamples.txt', 'w+')
for n in keptn:
    out.writelines(n+'\n')

out = open(cwd + '/data/human/outpops.txt', 'w+')
for n in keptpop:
    out.writelines(n+'\n')

out = open(cwd + '/data/human/outsuper.txt', 'w+')
for n in keptspop:
    out.writelines(n+'\n')
