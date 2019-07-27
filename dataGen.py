from utilsDb import processScan
import glob2
import os
import pandas as pd
import h5py
import pydicom
import numpy as np
from mask_functions import rle2mask
import cv2
import h5py
import tqdm

PATH_TRAIN = '..\\Data\\dicom-images-train\\'
PATH_TEST = '..\\Data\\dicom-images-test\\'

train = glob2.glob(os.path.join(PATH_TRAIN, '**/*.dcm'))
test = glob2.glob(os.path.join(PATH_TEST, '**/*.dcm'))

df = pd.read_csv('..\\Data\\train-rle.csv').set_index('ImageId')
idxs = set(df.index)

train_names = []
for f in train: #remove images without labels
    name = f.split('\\')[-1][:-4]
    if name in idxs: train_names.append(f)

trueOnly = True

IMG_SIZE = 1024
resize = (IMG_SIZE, IMG_SIZE)



h5path = '..\\out\\train_' + str(IMG_SIZE) + '.h5'
if trueOnly:
    h5path = '..\\out\\train_true_' + str(IMG_SIZE) + '.h5'



# Keeping only true names:

if trueOnly:
    SEGMENTATION = '..\\Data\\train-rle.csv'
    anns = pd.read_csv(SEGMENTATION)

    pneumothorax_anns = anns[anns.EncodedPixels != " -1"].ImageId.unique().tolist()

    true_names = []
    for f in train_names: #remove images without labels
        name = f.split('\\')[-1][:-4]
        if name in pneumothorax_anns: true_names.append(f)
    
    train_names = true_names

nimages = len(train_names)
shape = (nimages, IMG_SIZE, IMG_SIZE)

h5file = h5py.File(h5path, "w")
h5file.create_dataset("image", shape)
h5file.create_dataset("mask", shape)

didx = 0

print("Creating Dataset")

for scanPath in train_names:
    print("processing image: " + str(didx) + " of " + str(nimages))
    img, imgMask = processScan(scanPath, df, resize)
    h5file["image"][didx] = img
    h5file["mask"][didx] = imgMask
    didx = didx+1

h5file.close()

'''

resize = False
#resize = (1024, 1024)

name = train_names[200]

scan = pydicom.read_file(name)
img = scan.pixel_array

if resize:
    img = cv2.resize(img, resize)

i = 1007

#for i in range(999,1010):
name = train_names[i]
index = name.split('/')[-1][:-4]
masks = df.loc[index,'EncodedPixels']
#print(mask)
mask_count = 0

imgMask = np.zeros((1024,1024))
print(masks)

if(type(masks) != str or (type(masks) == str and masks != ' -1')):
    if(type(masks) == str): masks = [masks]
    # else: masks = masks.tolist()
    mask_count +=1
    for mask in masks:
        imgMask += rle2mask(mask, 1024, 1024).T

import matplotlib.pyplot as plt

plt.imshow(imgMask)
plt.show()

import matplotlib.pyplot as plt

'''