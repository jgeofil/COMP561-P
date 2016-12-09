
from cStringIO import StringIO

out = []
peeps = 1000
for i in range(peeps):
    out.append(StringIO())

#label = open('labels.txt')
#labels = label.readline().split()
#labels.append('??')
labels = [x for x in range(peeps)]
with open('data/SNPs_only2.recode.vcf') as f:
    lines = f.readlines()
    for l in lines:
        if l[0] != '#':
            s = l.split()
            if s[3] in ['A','T','G','C'] and s[4] in ['A','T','G','C']:
                for i, c in enumerate(s[9:9+peeps]):
                    c = c.split('|')[0]
                    if int(c) == 0:
                        cha = s[3]
                    else:
                        cha = s[4]
                    out[i].write(cha)

f = open('data/out2.fasta', 'w')

for i, seq in enumerate(out):
    f.write('>'+str(labels[i])+'\n'+seq.getvalue()+'\n')
f.close()
