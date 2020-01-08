from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pickle

def l1_loss_function(y_actual, y_predicted):
        result =  K.mean(K.abs(y_predicted - y_actual))
        f.write(str(result.shape))
        return result

def scaledMSE_loss_function(y_actual, y_predicted):
        return 1000.0*(K.mean(K.square(y_predicted - y_actual)))
        
def l3_loss_function(y_actual, y_predicted):
        return K.mean(K.pow(y_predicted - y_actual,3), axis=-1)

def l4_loss_function(y_actual, y_predicted):
        return K.mean(K.pow(y_predicted - y_actual,4), axis=-1)
def linf_loss_function(y_actual, y_predicted):
        return K.mean(K.maximum(y_predicted , y_actual), axis=-1)


def fftMSe_loss_function(y_actual, y_predicted):
	y_actual_mean = tf.reduce_mean(y_actual, 3)
	y_predicted_mean =  tf.reduce_mean(y_predicted, 3)
        y_act_fft =  tf.abs(tf.spectral.rfft2d(y_actual_mean,[256,256]))
        y_pred_fft =  tf.abs(tf.spectral.rfft2d(y_predicted_mean,[256,256]))
        result = (K.mean(K.square(y_pred_fft - y_act_fft)) *.5/1000 ) +  0.5*(K.mean(K.square(y_predicted - y_actual)))
	return  result

# the histogramMSE_loss_function is to be used with a batch size of 1 
def histogramMSE_loss_function(y_actual, y_predicted):

        histActual = []
        histPredicted = []
        value_range = [0.0, 1.0]
        for i in range(8):
                
                histActualChannel = tf.histogram_fixed_width(y_actual[0,:,:,i], value_range, nbins=8)
                histPredictedChannel = tf.histogram_fixed_width( y_predicted[0,:,:,i], value_range, nbins=8)
                histActual.append(histActualChannel)
                histPredicted.append(histPredictedChannel)
        histActTensor = tf.convert_to_tensor(histActual)
        histPredictedTensor = tf.convert_to_tensor(histPredicted)
        result=K.mean(K.square(y_predicted - y_actual))                 
	return  result  + K.mean(K.square(y_predicted - y_actual), axis=-1)
	
	
def getModel(img_width, img_height, channel):
	input_img = Input(shape=(img_width, img_height, channel))    # adapt this if using 'channels_first' image data format

	x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)   
	x = MaxPooling2D((2, 2), padding='same')(x)

	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)

	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)

	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)


	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	encoded = MaxPooling2D((2, 2), padding='same')(x)


	# at this point the representation is (4, 4, 8), i.e. 128-dimensional

	x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
	x = UpSampling2D((2, 2))(x)


	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)


	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)

	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)

	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)


	decoded = Conv2D(8, (3, 3), activation='sigmoid', padding='same')(x)

	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')    # autoencoder.compile(optimizer='adadelta', loss='kullback_leibler_divergence')   to use kld   & autoencoder.compile(optimizer='adadelta', loss='logcosh') to use logCosh
	print(autoencoder.summary())
	return autoencoder



