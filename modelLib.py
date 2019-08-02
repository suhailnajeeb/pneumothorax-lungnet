from keras.models import Model
#import customInitializer

# core layers
from keras.layers import Lambda, Input, Dense, Dropout, Flatten, Activation
from keras.layers.merge import add, concatenate, average, multiply
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU, ReLU
from keras.regularizers import l2
from keras.optimizers import Adam

from keras.losses import binary_crossentropy

# layers for 2d convnet
from keras.layers import Reshape, Conv2D, LocallyConnected2D, Conv2DTranspose, MaxPooling2D, UpSampling2D,  SpatialDropout2D, Cropping2D, ZeroPadding2D, GlobalMaxPooling2D

# layers of 3d convnet
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D,  AveragePooling2D, AveragePooling3D,  SpatialDropout3D, Cropping3D, ZeroPadding3D

# recurrent layers
from keras.layers import TimeDistributed, ConvLSTM2D

# other utils
from keras import backend as K
from keras.utils import plot_model
from keras.models import load_model
	
#import customLoss as cl

import sys

# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred = K.cast(y_pred, 'float32')
#     y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
#     intersection = y_true_f * y_pred_f
#     score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
#     return score

# def dice_loss(y_true, y_pred):
#     smooth = 1.
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = y_true_f * y_pred_f
#     score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#     return 1. - score

# def bce_dice_loss(y_true, y_pred):
#     return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# def bce_logdice_loss(y_true, y_pred):
#     return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))

# def log_dice_coef_loss(y_true, y_pred):
# 	''' This function calculates the and returns the negative of log of the 
# 	dice coefficient between the tensors y_true (ground truth) and y_pred (model output) '''
# 	return - K.log(dice_coef(y_true,y_pred) + 1e-8)


# # original one, not working

# def dice_coef(y_true, y_pred, smooth=1):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dummynet():
    inputs = Input((256, 256, 1))
    cc = Conv2D(1, (3, 3), activation='relu', padding='same') (inputs)
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (cc)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
    model.summary()
    return model

def LungNet001a():
	
	'''Change Log :
	added bottleneck after concatentation
	increase no. of filters
	decreased dropout to 0.4
	'''

	def bn_block(x):
		return add([x, BatchNormalization()(x)])

	def conv_block(x, nb_filter, filter_size, atrous_rate=(1, 1)):
		x = Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same')(x)
		x = bn_block(x)
		x = ReLU()(x)
		return x


	atrous_rates = [(1,1), (1,1), (2,2), (3,3), (5,5), (8,8), (13,13), (21,21), (34,34), (55,55)] 
		
	i = Input((512, 512, 1))
	
	t = bn_block(i)
	feat_list = [t]
	for layer in range(0, len(atrous_rates)):
		t = conv_block(t, 32, (3,3), atrous_rate=atrous_rates[layer])
		feat_list.append(t)
		
	t = concatenate(feat_list)
	t = Dropout(0.4)(t)

	t = conv_block(t, 128, (1, 1))
	t = conv_block(t, 64, (1, 1))
	t = conv_block(t, 128, (1, 1))
	t = conv_block(t, 32, (1,1))

	t = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal')(t)

	return Model(inputs=[i], outputs=[t])

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def basic_unet(img_rows=256, img_cols=256):
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model
