import os
import subprocess
import glob

cwd = os.getcwd()

files = glob.glob('data/human/chr1.vcf.gz')
print files

for f in files:
    fn = f.split('/')[-1].split('.')[0]
    bashCommand = 'vcftools --gzvcf '+cwd+'/'+str(f)+' --out '+cwd+'/data/human/'+str(fn)+' --keep '+cwd+'/data/human/outsamples.txt --mac 2 --max-missing 1 --remove-indels --min-alleles 2 --max-alleles 2 --recode'
    print bashCommand

    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print output
    print error
