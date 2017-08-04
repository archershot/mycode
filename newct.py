from pycocotools import coco
from color_transfer import color_transfer
import os
import random
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--batchid', type=int,required=True, help='input batch id from 1~8')
args = parser.parse_args()

env_dist = os.environ
anndata = coco.COCO(env_dist.get('MSCOCO_ANN_PATH')+'/instances_train2014.json')
print ("Load annotation file complete!")

imgids = anndata.getImgIds()
imgids.sort()
imgsize = len(imgids)

startpos = imgsize * (args.batchid-1) / 8
endpos = imgsize * (args.batchid) / 8
print ('imgsize:%d,batchid:%d batchrange:(%d,%d)'%(imgsize,args.batchid,startpos,endpos))
for imgid in imgids[startpos:endpos]:
    annids = anndata.getAnnIds(imgIds=[imgid],catIds=[1])
    if len(annids) == 0:
	continue
    masksum = 0
    for annid in annids:
        ann = anndata.loadAnns(ids=annid)[0]
        curmask = anndata.annToMask(ann)
        ms = curmask.sum()
        if ms>masksum:
            masksum = ms
            mask = curmask.copy()
    if masksum <= 222:
        continue
    timgann = anndata.loadImgs(imgid)[0]
    timg = cv2.imread(env_dist.get('MSCOCO_IMG_PATH')+'/'+timgann['file_name'])
    simgann = anndata.loadImgs(random.choice(imgids))[0]
    simg = cv2.imread(env_dist.get('MSCOCO_IMG_PATH')+'/'+simgann['file_name'])
    resimg = color_transfer(simg,timg)
    maskidx = (mask==0)
    resimg[maskidx] = timg[maskidx]
    cv2.imwrite(env_dist.get('MSCOCO_PATH')+'/newdata/gt/'+str(timgann['id'])+'.png',timg)
    cv2.imwrite(env_dist.get('MSCOCO_PATH')+'/newdata/image/'+str(timgann['id'])+'.png',resimg)
    cv2.imwrite(env_dist.get('MSCOCO_PATH')+'/newdata/mask/'+str(timgann['id'])+'.png',mask*255)
    print imgid
print maxann
