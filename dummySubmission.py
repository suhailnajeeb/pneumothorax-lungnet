import pandas as pd
import glob2
import os

PATH_VAL = '..\\Data\\dicom-images-test\\'
SUBCSV = '..\\out\\submission.csv'

val = glob2.glob(os.path.join(PATH_VAL,'**/*.dcm'))

ids = []
rles = []

for f in val:
    id = f.split('\\')[-1][:-4]
    rle = ''
    ids.append(id)
    rles.append(rle)

sub_df = pd.DataFrame({'ImageId': ids, 'EncodedPixels': rles})
sub_df.loc[sub_df.EncodedPixels=='', 'EncodedPixels'] = '-1'
sub_df.head()

sub_df.to_csv(SUBCSV, index=False)