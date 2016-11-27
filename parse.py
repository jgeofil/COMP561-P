
from cStringIO import StringIO

out = []
for i in range(83):
    out.append(StringIO())

label = open('labels.txt')
labels = label.readline().split()
labels.append('??')

with open('in.txt') as f:
    lines = f.readlines()
    for l in lines:
        if l[0] != '#':
            s = l.split()
            for i, c in enumerate(s[9:]):
                #print l[3], l[4]
                if int(c) == 0:
                    cha = s[3]
                else:
                    cha = s[4]
                out[i].write(cha)

f = open('out.fasta', 'w')

for i, seq in enumerate(out):
    f.write('>'+labels[i]+'\n'+seq.getvalue()+'\n')
f.close()