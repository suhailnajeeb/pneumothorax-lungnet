#from modelLib import dummynet
from modelLib import LungNet001a, basic_unet, dice_coef, dice_coef_loss
from utilsTrain import generator

import h5py
from sklearn.model_selection import train_test_split
import numpy as np
import os

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau
from keras.utils.training_utils import multi_gpu_model


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto() 
# dynamically grow GPU memory 
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

batchSize = 4
NO_GPU = 4

h5path = "..\\out\\train_true_256.h5"
h5file = h5py.File(h5path, "r")

n = h5file["image"].shape[0]
a = np.arange(n)

train_index, test_index = train_test_split(a, test_size=0.2, random_state=42)

trainGen = generator(h5file, train_index, batchSize)
testGen = generator(h5file, test_index, batchSize)


weightsFolder = '..\\out\\weights\\'
modelName = 'LungNet'
bestModelPath = '..\\out\\weights\\best.hdf5'
modelFolder = '..\\out\\model\\'

epochs = 100
epochStart = 0

patience = 50

# Define the Model

#model = dummynet()
model = LungNet001a()
#model = multi_gpu_model(model, gpus = NO_GPU)
model = basic_unet()

# Compile the Model & Configure

model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef])
model.summary()

# Fit the Model
# x,y = next(trainGen)
# model.fit(x,y) # Use this line to check if the model is compiling and ignore others

check1 = ModelCheckpoint(os.path.join(weightsFolder, modelName + "_{epoch:02d}-loss-{val_loss:.3f}.hdf5"), monitor='val_loss', save_best_only=True, mode='auto')
check2 = ModelCheckpoint(bestModelPath, monitor='val_loss', save_best_only=True, mode='auto')
check3 = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=patience, verbose=0, mode='auto')
check4 = CSVLogger(os.path.join(modelFolder, modelName +'_trainingLog.csv'), separator=',', append=True)
check5 = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience//1.5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=1e-10)


trained_model = model.fit_generator(trainGen, steps_per_epoch=(len(train_index) // batchSize), epochs=epochs, initial_epoch=epochStart,
											validation_data= testGen, validation_steps=(len(test_index) // batchSize), callbacks=[check1,check2,check3,check4,check5], 
											verbose=1)

#model.fit_generator(train_generator, test_generator, )

# Plot metrics 

# cleanup

trainGen.close()
testGen.close()
h5file.close()

''' kahini with the optimizer

