import os
import shutil
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot',type=str,required=True)
args = parser.parse_args()

testdir = os.path.join(args.dataroot,'test')

testfilenames = sorted(os.listdir(os.path.join(testdir, 'image')))

with open('testlist', "w") as logfile:
    for x in testfilenames:
        logfile.write(x+'\n')
