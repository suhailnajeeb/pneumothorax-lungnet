from keras.models import Model
#import customInitializer

# core layers
from keras.layers import Lambda, Input, Dense, Dropout, Flatten, Activation
from keras.layers.merge import add, concatenate, average, multiply
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU, ReLU
from keras.regularizers import l2

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

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

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
		
	i = Input((256, 256, 1))
	
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