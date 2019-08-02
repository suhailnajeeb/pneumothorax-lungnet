import pandas as pd
import glob2
import os

PATH_VAL = '..\\Data\\dicom-images-test\\'
CSVFILE = '..\\Data\\kanvari.csv'
SUBCSV = '..\\out\\submission.csv'

val = glob2.glob(os.path.join(PATH_VAL,'**/*.dcm'))

df = pd.read_csv('..\\Data\\kanvari.csv')

ids = []
rles = []


# Issue: f = '1.2.276.0.7230010.3.1.4.8323329.7020.1517875202.386064'

for f in val:
    id = f.split('\\')[-1][:-4]
    x = df.loc[df.ImageId == id]
    try:
        x = x.iloc[0]['EncodedPixels']
        if (x == -1):
            rle = '-1'
        else:
            rle = '0'
    except:
        rle = '0'
    ids.append(id)
    rles.append(rle)

sub_df = pd.DataFrame({'ImageId': ids, 'EncodedPixels': rles})
sub_df.head()

sub_df.to_csv(SUBCSV, index=False)