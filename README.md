# AutoEncoder
Autoencoder for compressing multi-spectral satellite imagery
# Installation 
Clone the git project:

```
$ git clone https://github.com/wisekrack/AutoEncoder.git
```
Tested on Python2 , make changes in requirements.txt if you are using Python3
The requirements (including tensorflow) can be installed using:
```
pip install -r requirements.txt
```

## Train the model

Train the model on the 4 tile data
```
python trainAutoencoder.py
```
Please note that you can change the loss function according to your need from the model.py , the loss functions are written on top and necessary instructions provided. 

## Results
Inorder to view the original image in RGB , use the following function
```
python viewImages.py 
```
Inorder to view the inference results , will provide you the RGB inferences of the autoencoder.
```
python inference.py
```


WiseKrack 
