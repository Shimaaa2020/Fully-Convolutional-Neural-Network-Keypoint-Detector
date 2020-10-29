import keras

Conv2D = keras.layers.Conv2D

def my_cnn(input0, input1, x_train, y_train, mask, epochs=10, batch_size=1, lr=0.01, loadWeights=False, saveWeights=False):

    net = Conv2D(8, (3, 3), padding='same', activation='relu')(input0)
    net = Conv2D(16, (3, 3), padding='same', activation='relu')(net)
    net = Conv2D(32, (3, 3), padding='same', activation='relu')(net)
    net = Conv2D(64, (3, 3), padding='same', activation='relu')(net)
    net = Conv2D(128, (3, 3), padding='same', activation='relu')(net)
    net = Conv2D(256, (3, 3), padding='same', activation='relu')(net)
    net = Conv2D(64, (3, 3), padding='same', activation='relu')(net)
    net = Conv2D(16, (3, 3), padding='same', activation='relu')(net)
    net = Conv2D(4, (3, 3), padding='same', activation='relu')(net)
    output = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(net)
    
    model = keras.models.Model(inputs=[input0, input1], outputs=[output])
    if loadWeights:
        loadWeightsFile = '/home/mehrdad/sample/mdl_1500.hdf5'
        model.load_weights(loadWeightsFile, by_name=False)
        
    return model

