import os
import subprocess
import glob

cwd = os.getcwd()

files = glob.glob('data/human/out*.nex')
print files
for f in range(len(files)):
    bashCommand = '/home/jay/splitstree4/SplitsTree -g -i '+cwd+'/'+'data/human/out'+str(f)+'.nex'
    print bashCommand

    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print output
    print error
