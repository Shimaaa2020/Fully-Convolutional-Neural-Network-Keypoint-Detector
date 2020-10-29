import keras

def masked_loss(mask):
    def loss(y_true, y_pred):
        diff = (y_pred - y_true) * mask
        return keras.backend.mean(keras.backend.square(diff), axis=-1)
    return loss
