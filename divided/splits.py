import os
import subprocess
import glob

cwd = os.getcwd()

files = glob.glob('data/divided/out*.nex')
print files
for f in files:
    bashCommand = '/home/jay/splitstree4/SplitsTree -g -i '+cwd+'/'+f
    print bashCommand

    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print output
    print error
