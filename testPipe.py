from modelLib import basic_unet
from sklearn.model_selection import train_test_split
from utilsTrain import generator
from modelLib import dice_coef
import h5py
import numpy as np

batchSize = 1

h5path = "..\\out\\train_true_256.h5"
h5file = h5py.File(h5path, "r")

n = h5file["image"].shape[0]
a = np.arange(n)

train_index, test_index = train_test_split(a, test_size=0.2, random_state=42)

trainGen = generator(h5file, train_index, batchSize)
testGen = generator(h5file, test_index, batchSize)

model = basic_unet()

bestModelPath = '..\\out\\weights\\UNET_01-loss--1.375.hdf5'

model.load_weights(bestModelPath)

v = next(trainGen)

x = v[0]
y_true = v[1]
y_pred = model.predict(x)

y_true = y_true.flatten()
y_pred = y_pred.flatten()

intersection = y_true*y_pred
sum(intersection)

import matplotlib.pyplot as plt
plt.imshow(y_pred[0][:,:,0], cmap = 'gray')
plt.show()

plt.imshow(y_true[0][:,:,0], cmap = 'gray')
plt.show()