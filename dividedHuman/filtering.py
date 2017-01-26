import os
import subprocess
import glob

cwd = os.getcwd()

files = glob.glob('data/human/chr*.vcf.gz')
print files
for f in files:
    bashCommand = 'vcftools --gzvcf '+cwd+'/'+str(f)+' --out '+cwd+'/'+str(f)+'.filtered --keep '+cwd+'/data/human/outsamples.txt --maf 0.0007 --max-missing 1 --remove-indels --min-alleles 2 --max-alleles 2 --recode'
    print bashCommand

    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print output
    print error
