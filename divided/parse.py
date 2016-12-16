
from cStringIO import StringIO
import os
cwd = os.getcwd()

label = open('labels.txt')
labels = label.readline().split()
labels.append('??')

SIZE = len(labels)
DATA = 'data/divided/'

def stringIOArray(size):
    out = []
    for i in range(size):
        out.append(StringIO())
    return out

def writeHeader(f, name, numTaxa, numChar):
    f.write('#nexus\n')
    f.write('['+str(name)+']\n')
    f.write('begin taxa;\n')
    f.write('dimensions ntax='+str(numTaxa)+';\n')
    f.write('end;\n\n')
    f.write('begin characters;\n')
    f.write('dimensions nchar='+str(numChar)+';\n')
    f.write('format datatype=dna;\n')
    f.write('matrix\n')

def writeFooter(f,i):
    f.write(';\nend;\n')
    f.write('BEGIN st_Assumptions;\n')
    f.write('\tchartransform=Uncorrected_P;\n')
    f.write('\tdisttransform=NeighborNet;\n')
    f.write('\tsplitstransform=EqualAngle;\n')
    f.write('\tautolayoutnodelabels;\n')
    f.write('END; [st_Assumptions]\n\n')
    f.write('begin SplitsTree;\n')
    f.write('\tUPDATE;\n')
    f.write('\tEXPORT FILE='+cwd+'/'+DATA+'splits'+str(i)+'.nex REPLACE=yes;\n')
    f.write('\tQUIT;\n')
    f.write('end;\n')


contigs = []

with open(DATA+'in.txt') as f:
    strio = stringIOArray(SIZE)

    lines = f.readlines()
    lastContig = 'scaffold_1'
    count = 0
    for l in lines:

        if l[0] != '#':
            s = l.split()
            if s[0] == 'scaffold_1':
                count +=1
            if s[0] == lastContig:
                for i, c in enumerate(s[9:]):
                    if int(c) == 0:
                        cha = s[3]
                    else:
                        cha = s[4]
                    strio[i].write(cha)
            else:
                print count
                lastContig = s[0]
                print lastContig
                contigs.append(strio)
                strio = stringIOArray(SIZE)
    contigs.append(strio)

for i, cont in enumerate(contigs):
    print i
    f = open(DATA+'out'+str(i)+'.nex', 'w')

    for j, seq in enumerate(cont):
        cont[j] = seq.getvalue()

    writeHeader(f,i,SIZE, len(cont[0]))
    for j, seq in enumerate(cont):
        f.write(str(labels[j])+'\t'+seq+'\n')
    writeFooter(f,i)
    f.close()
