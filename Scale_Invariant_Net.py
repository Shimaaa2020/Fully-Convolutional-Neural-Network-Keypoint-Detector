import keras
from my_cnn import my_cnn
from readCSV import load2d
from Confidence_Map import confidence_map
from masked_loss import masked_loss

Conv2D = keras.layers.Conv2D

if __name__=="__main__" :
    batch_size = 1
    epochs = 1
    lr = 0.01

    x_train, y = load2d()
    mask, y_train = confidence_map(x_train, y)

    input0 = keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    input1 = keras.layers.Input(shape=(y_train.shape[1], y_train.shape[2], y_train.shape[3]))
    
    model = my_cnn(input0, input1, x_train, y_train, mask, epochs=epochs, batch_size=batch_size)
    
    model.compile(loss=masked_loss(mask=mask) , optimizer=keras.optimizers.SGD(lr=lr))
    model.fit([x_train, mask],[y_train], batch_size=batch_size, epochs=epochs, verbose=1)
    
    saveWeightsFile = '/home/mehrdad/sample/mdl_2500.hdf5
