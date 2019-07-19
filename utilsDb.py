import pydicom
import cv2
from mask_functions import rle2mask
import numpy as np

def processScan(name,df,resize = (1024,1024), ORIG_DIM = 1024):
    scan = pydicom.read_file(name)
    img = scan.pixel_array
    
    if (resize != (ORIG_DIM,ORIG_DIM)):
        img = cv2.resize(img, resize)

    index = name.split('/')[-1][:-4]
    masks = df.loc[index,'EncodedPixels']
    mask_count = 0

    imgMask = np.zeros((ORIG_DIM,ORIG_DIM))

    if(type(masks) != str or (type(masks) == str and masks != ' -1')):
        if(type(masks) == str): masks = [masks]
        # else: masks = masks.tolist()
        mask_count +=1
        for mask in masks:
            imgMask += rle2mask(mask, ORIG_DIM, ORIG_DIM).T

    if (resize != (ORIG_DIM,ORIG_DIM)):
        imgMask = cv2.resize(imgMask, resize)

    return img, imgMask
