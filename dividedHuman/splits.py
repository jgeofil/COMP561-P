import os
import subprocess
import glob

cwd = os.getcwd()

files = glob.glob('data/human/chr2.nex')
print files
for f in files:
    bashCommand = '/home/jay/splitstree4/SplitsTree -g -m -v -i '+cwd+'/'+str(f)
    print bashCommand

    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print output
    print error
