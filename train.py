from modelLib import dummynet
from utilsTrain import generator


import h5py
from sklearn.model_selection import train_test_split
import numpy as np

h5path = "../out/train.h5"
h5file = h5py.File(h5path, "r")

n = h5file["image"].shape[0]
a = np.arange(n)

train, test = train_test_split(a, test_size=0.2, random_state=42)

train_generator = generator(h5file, train, 32)
test_generator = generator(h5file, test, 32)

x,y = next(train_generator)

# Define the Model

model = dummynet()

# Compile the Model & Configure

# Fit the Model

model.fit(x,y)

# Plot metrics 

h5file.close()