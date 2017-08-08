from pycocotools import coco
import os
import random
import cv2
import numpy as np
    
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def image_stats(image, mask):
	# compute the mean and standard deviation of each channel
	(l, a, b) = cv2.split(image)
	maskidx = (mask == 1)
	(l, a, b) = (l[maskidx], a[maskidx], b[maskidx])
	(lMean, lStd) = (l.mean(), l.std())
	(aMean, aStd) = (a.mean(), a.std())
	(bMean, bStd) = (b.mean(), b.std())

	# return the color statistics
	return (lMean, lStd, aMean, aStd, bMean, bStd)

def color_transfer(simg, smask, timg, tmask):

	# convert the images from the RGB to L*ab* color space, being
	# sure to utilizing the floating point data type (note: OpenCV
	# expects floats to be 32-bit, so use that instead of 64-bit)
	source = cv2.cvtColor(simg, cv2.COLOR_BGR2LAB).astype("float32")
	target = cv2.cvtColor(timg, cv2.COLOR_BGR2LAB).astype("float32")

	# compute color statistics for the source and target images
	(lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source, smask)
	(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target, tmask)

	# subtract the means from the target image
	(l, a, b) = cv2.split(target)
	
	if lStdTar != 0 and lStdSrc != 0:
	    l = (lStdTar / lStdSrc) * (l - lMeanTar) + lMeanSrc
	    
	if aStdTar != 0 and aStdSrc != 0:
	    a = (aStdTar / aStdSrc) * (a - aMeanTar) + aMeanSrc
	    
	if bStdTar != 0 and bStdSrc != 0:
	    b = (bStdTar / bStdSrc) * (b - bMeanTar) + bMeanSrc

	# clip the pixel intensities to [0, 255] if they fall outside
	# this range
	l = np.clip(l, 0, 255)
	a = np.clip(a, 0, 255)
	b = np.clip(b, 0, 255)

	# merge the channels together and convert back to the RGB color
	# space, being sure to utilize the 8-bit unsigned integer data
	# type
	transfer = cv2.merge([l, a, b])
	transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
	
	umaskidx = (tmask == 0)
	transfer[umaskidx] = timg[umaskidx]
	
	# return the color transferred image
	return transfer
    
annpath = '/home/archer/dataset/mscoco/annotations/instances_train2014.json'
imgdir = '/home/archer/dataset/mscoco/train2014'
resdir = '/home/archer/dataset/mscoco/result'

mkdir(resdir)
mkdir(os.path.join(resdir,'gt'))
mkdir(os.path.join(resdir,'mask'))
mkdir(os.path.join(resdir,'image'))

annotations = coco.COCO(annpath)
print ("Load annotation file complete!")

imgids = sorted(annotations.getImgIds(catIds=1))
imgsize = len(imgids)

print ('imgsize:%d' % (imgsize))

validannimgids = []
for i, imgid in enumerate(imgids):
    annids = annotations.getAnnIds(imgIds=imgid, catIds=1)
    if len(annids) == 0:
	    continue
    maxmasksum = 0.0
    ansannid = 0
    for annid in annids:
        ann = annotations.loadAnns(ids=annid)[0]
        curmask = annotations.annToMask(ann)
        ms = 1.0 * curmask.sum().sum() / curmask.size
        if ms>maxmasksum:
            maxmasksum = ms
            ansannid = annid
    if maxmasksum < 0.1 :
        continue
        
print ('%d valid image for transfer' % len(validannimgids) )

totaliter = 0

for timgid,tannid in validannimgids:
    
    tmask = annotations.annToMask(annotations.loadAnns(ids=tannid)[0])
    timg = cv2.imread(os.path.join(imgdir, annotations.loadImgs(timgid)[0]['file_name']))
    
    (simgid,sannid) = random.choice(validannimgids)    
    smask = annotations.annToMask(annotations.loadAnns(ids=sannid)[0])
    simg = cv2.imread(os.path.join(imgdir, annotations.loadImgs(simgid)[0]['file_name']))
    
    resimg = color_transfer(simg, smask, timg, tmask)
    
    imgname = ('%.7d.png' % timgid)
    
    cv2.imwrite(os.path.join(resdir,'gt',imgname), timg)
    cv2.imwrite(os.path.join(resdir,'image',imgname), resimg)
    cv2.imwrite(os.path.join(resdir,'mask',imgname), tmask*255)
    
    totaliter += 1
    
    if totaliter % 500 == 0:
        print totaliter

    
    
    
    
    
    
    
    
    
