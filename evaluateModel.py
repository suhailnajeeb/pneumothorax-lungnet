import os
import glob2
import pandas as pd
from modelLib import LungNet001a
import pydicom
import cv2
import numpy as np
from mask_functions import mask2rle

def predictImage(f,IMG_SIZE = 512):
    ORIG_DIM = 1024
    resize = (IMG_SIZE, IMG_SIZE)

    scan = pydicom.read_file(f)
    img = scan.pixel_array

    if (resize != (ORIG_DIM,ORIG_DIM)):
        img = cv2.resize(img, resize)

    img = img/255
    img = np.expand_dims(img, axis = 2)
    img = np.expand_dims(img, axis = 0)
    mask = model.predict(img)

    mask = mask.reshape(512,512)
    mask = cv2.resize(mask, (1024,1024))
    mask = mask.round()
    mask = mask*255

    rle = mask2rle(mask, 1024, 1024)
    return rle

bestModelPath = '..\\out\\weights\\best.hdf5'

#model = load_model(bestModelPath)
model = LungNet001a()
model.load_weights(bestModelPath)

PATH_VAL = '..\\Data\\dicom-images-test\\'
CSVFILE = '..\\Data\\kanvari.csv'
SUBCSV = '..\\out\\submission.csv'

val = glob2.glob(os.path.join(PATH_VAL,'**/*.dcm'))

df = pd.read_csv('..\\Data\\kanvari.csv')

ids = []
rles = []

n = len(val)
idx = 0

for f in val:
    id = f.split('\\')[-1][:-4]
    x = df.loc[df.ImageId == id]
    try:
        x = x.iloc[0]['EncodedPixels']
        if (x == -1):
            print(id + " : False, Skipping " + str(idx) + " of " + str(n))
            rle = '-1'
        else:
            print(id + " : True, Predicting " + str(idx) + " of " + str(n))
            rle = predictImage(f)
    except:
        print(id + " : True, Predicting " + str(idx) + " of " + str(n))
        rle = predictImage(f)
    ids.append(id)
    rles.append(rle)
    idx = idx + 1

sub_df = pd.DataFrame({'ImageId': ids, 'EncodedPixels': rles})
sub_df.head()

sub_df.to_csv(SUBCSV, index=False)



#f = val[0]

#IMG_SIZE = 512

