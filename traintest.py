import os
import shutil
import random
import argparse

parser = argparse.ArgumentParse()
parser.add_argument('--dataroot',type=str,required=True)
parser.add_argument('--output',type=str,required=True)
args = parser.parse_args()

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

oridir = args.dataroot
traindir = os.path.join(args.output,'train')
testdir = os.path.join(args.output,'test')
#oridir = '/home/archer/dataset/mscoco/result'
#traindir = '/home/archer/dataset/mscoco/train'
#testdir = '/home/archer/dataset/mscoco/test'

mkdir(traindir)
mkdir(os.path.join(traindir,'gt'))
mkdir(os.path.join(traindir,'mask'))
mkdir(os.path.join(traindir,'image'))

mkdir(testdir)
mkdir(os.path.join(testdir,'gt'))
mkdir(os.path.join(testdir,'mask'))
mkdir(os.path.join(testdir,'image'))

filenames = sorted(os.listdir(os.path.join(oridir,'image')))

totalsize = len(filenames)
testsize = totalsize * 7 / 100

testidx = random.sample(range(totalsize),testsize)
testmask = [False] * totalsize
for x in testidx:
    testmask[x] = True
    
for idx in range(totalsize):
    resdir = testdir if testmask[idx] else traindir
    shutil.move(os.path.join(oridir,'gt',filenames[idx]), os.path.join(resdir,'gt',filenames[idx])) 
    shutil.move(os.path.join(oridir,'mask',filenames[idx]), os.path.join(resdir,'mask',filenames[idx])) 
    shutil.move(os.path.join(oridir,'image',filenames[idx]), os.path.join(resdir,'image',filenames[idx])) 
    if (idx+1) % 1000 == 0:
        print idx+1

