import dataLoader
import model
from keras.models import model_from_json
from keras.callbacks import TensorBoard
import os 
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
processedData, unprocessedData = dataLoader.fetch_data('data/')


autoencoder = model.getModel(256, 256, 8)

x_train = processedData
x_test = processedData[1,:,:,:].reshape(1,256,256,8)
autoencoder.fit(x_train, x_train, epochs=100000, batch_size=4, shuffle=True, validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='conv_autoencoder')], verbose=2)
                
autoencoder_json = autoencoder.to_json()
with open("model.json", "w") as json_file:
    json_file.write(autoencoder_json)
# serialize weights to HDF5
autoencoder.save_weights("model.h5")
print("Saved model to disk")
autoencoder.save('autoencoder.h5')
