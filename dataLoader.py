import glob 
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

def fetch_data(folderPath):
	processedData = np.array([], dtype = np.float32).reshape(0,256,256,8)
	unprocessedData = np.array([], dtype = np.float32).reshape(0,256,256,8)
	for tilePath in glob.glob(folderPath+'/*.tif'):

		tile = tiff.imread(tilePath)
		tile = tile.reshape(1,tile.shape[0],tile.shape[1],tile.shape[2])
		tile = tile.astype('float32')
		unprocessedData = np.vstack((unprocessedData,tile))
		print(tile.shape, np.min(tile), np.max(tile))   #clearly the data has wide variance, we need  to normalize the data
		for channelNum in range(tile.shape[3]):
			spectralImage = tile[:,:,:,channelNum]
#			print(spectralImage.shape,  spectralImage.max(), spectralImage.min())
			spectralImage = (( spectralImage - spectralImage.min())*1.00) / ( (spectralImage.max() -spectralImage.min() )*1.0)
#			print(spectralImage.shape,  spectralImage.max(), spectralImage.min())
#			plt.imshow((spectralImage.reshape(256,256))*255)
#			plt.show()
			tile[:,:,:,channelNum] = spectralImage
#			plt.imshow(tile[0,:,:,channelNum]*255)
#			plt.show()
#		print(tile.shape, np.argmin(tile), np.argmax(tile))   #clearly the data has wide variance, we need  to normalize the data
		processedData = np.vstack((processedData,tile))

	return processedData, unprocessedData
#data1, data2=fetch_data('data/')
#print(data1.shape, data2.shape)

def unprocessData(processedData, unprocessedData):
        processedUndone = processedData
        for tileNumber in range(processedData.shape[0]):
                sample = processedData[tileNumber,:,:,:]
                for channel in range(sample.shape[2]):
                        channelMin = unprocessedData[tileNumber, : , : , channel ].min()
                        channelMax = unprocessedData[tileNumber, : , : , channel ].max()
                        print(channelMax, channelMin)
                        processedSpectral = ( processedData[tileNumber, : , : , channel ] *(channelMax - channelMin))  +  np.array(channelMin)
                        processedUndone[tileNumber, : , : , channel] = processedSpectral
        print(np.sum(np.absolute( processedUndone  -  unprocessedData  )))
        return processedUndone

#data1, data2=fetch_data('data/')
#data2Sharp = unprocessData(data1, data2)
