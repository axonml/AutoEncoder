import glob 
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2 

def viewRGB(folderPath):
        count = 0
	for tilePath in glob.glob(folderPath+'/*.tif'):
                count += 1
		tile = tiff.imread(tilePath)
		for spectra in range(tile.shape[2]):
		        print(tile[:,:,spectra].max(), tile[:,:,spectra].min()   )
                rgb = tile[:,:,[1,2,4]] *0.8
#                rgb = 
                cv2.imwrite(str(count)+'.jpg', rgb)
                print(rgb.shape)
                plt.imshow(rgb)
#                plt.show()
                
viewRGB('data/')
