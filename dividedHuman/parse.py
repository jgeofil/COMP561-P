
from cStringIO import StringIO
import os
import glob
cwd = os.getcwd()

label = open('data/human/outsamples.txt')

labels =[]
for l in label.readlines():
    labels.append(l.rstrip())

print labels
SIZE = len(labels)
DATA = 'data/human/'

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

def writeFooter(f,i,fn):
    filename = str(fn).split('/')[-1].split('.')[0]
    f.write(';\nend;\n')
    f.write('BEGIN st_Assumptions;\n')
    f.write('\tchartransform=Uncorrected_P;\n')
    f.write('\tdisttransform=NeighborNet;\n')
    f.write('\tsplitstransform=EqualAngle;\n')
    f.write('\tautolayoutnodelabels;\n')
    f.write('END; [st_Assumptions]\n\n')
    f.write('begin SplitsTree;\n')
    f.write('\tUPDATE;\n')
    f.write('\tEXPORT FILE='+cwd+'/'+DATA+'splits.'+str(filename)+'.nex REPLACE=yes;\n')
    f.write('\tQUIT;\n')
    f.write('end;\n')


contigs = []

files = glob.glob('data/human/chr22.recode.vcf')
print files

for fix,f in enumerate(files):
    filename = str(f).split('/')[-1].split('.')[0]
    with open(f) as f:
        strio = stringIOArray(SIZE)

        print('Parsing '+str(f))
        lines = f.readlines()
        for l in lines:
            if l[0] != '#':
                s = l.split()

                for i, c in enumerate(s[9:]):
                    if int(c[0]) == 0:
                        cha = s[3]
                    else:
                        cha = s[4]
                    if int(c[2]) == 0:
                        cha2 = s[3]
                    else:
                        cha2 = s[4]
                    strio[i].write(cha)
                    strio[i].write(cha2)

        print('Writing '+str(f))
        fo = open(DATA+filename+'.nex', 'w')

        for j, seq in enumerate(strio):
            strio[j] = seq.getvalue()

        writeHeader(fo,i,SIZE, len(strio[0]))
        for j, seq in enumerate(strio):
            fo.write(str(labels[j])+'\t'+seq+'\n')
        writeFooter(fo,fix,f)
        fo.close()

        strio = stringIOArray(SIZE)
