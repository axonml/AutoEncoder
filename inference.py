import dataLoader
import numpy as np
import model
from numpy import loadtxt
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import tifffile as tiff
processedData, unprocessedData = dataLoader.fetch_data('data/')

autoencoder = model.getModel(256, 256, 8)
autoencoder.load_weights('model.h5')
# summarize model.
autoencoder.summary()
predictedDataStore = np.array([], dtype = np.float32).reshape(0,256,256,8)
print(processedData.shape)
raw_input()
for a in range(processedData.shape[0]):
        predictedTest = autoencoder.predict(processedData[a,:,:,:].reshape(1,256,256,8))
        result = np.sum(np.absolute(predictedTest - processedData[1,:,:,:].reshape(1,256,256,8)))
        print(result)
        predictedDataStore = np.vstack((predictedDataStore, predictedTest))
        
        for i in range(8):
                sample = processedData[a,:,:,i]
                print(sample.shape)
                plt.imshow((sample.reshape(256,256))*255.0)
#                plt.show()
                cv2.imwrite(str(a)+'Tile'+str(i)+'.jpg', (sample.reshape(256,256))*255.0   )
                plt.imshow((predictedTest[0,:,:,i].reshape(256,256))*255.0)
#                plt.show()
                cv2.imwrite(str(a)+'Costructed'+str(i)+'.jpg', (predictedTest[0,:,:,i].reshape(256,256))*255.0   )
predictedCorrected = dataLoader.unprocessData(predictedDataStore, unprocessedData)
print(np.sum(np.absolute( predictedCorrected  -  unprocessedData  )))
for a in range(predictedCorrected.shape[0]):
        tilePredicted = predictedCorrected[a,:,:,:]
        predictedRGB = tilePredicted[:,:,[1,2,4]]
        cv2.imwrite(str(a)+'predicted.png',predictedRGB)
        tiff.imwrite(str(a)+'.tif', tilePredicted)
