import keras
import numpy as np
from my_cnn import my_cnn
from readCSV import load2d
from Confidence_Map import confidence_map
from matplotlib import pyplot as pp
pp.ion()

global mask
global target
global pred
global x_test

def load_model():
    filepath = '/home/mehrdad/sample/mdl_1125.hdf5'
    x, y = load2d()
    x_test, _ = load2d(test=True)
    mask, target = confidence_map(x, y)
    input0 = keras.layers.Input(shape=(None, None, 1))
    input1 = keras.layers.Input(shape=(None, None, 1))

    model = my_cnn(input0, input1, x_test, target, mask)
    model.load_weights(filepath, by_name=False)
    
    
    return mask, target, model, x_test

def plot_target(target):
    pp.matshow(target[0, :, :, 0])
    pp.colorbar()

def plot_model_prediction(model, x_test, mask):
    pred = model.predict([x_test, mask])
    fig = pp.figure(figsize=(9, 9))
    rand_num = np.random.randint(0, pred.shape[0])
    for i in range(rand_num, rand_num + 9):
        fig.add_subplot(3, 3, i - rand_num + 1)
        pp.matshow(pred[i, :, :, 0], fignum=False)

def plot_scale_test(mask, model, x_test):        
    for i in range(3):
        pic_size = np.random.randint(96, 256)
        Pic = np.ones((pic_size, pic_size))
        pic_position = np.random.randint(0, pic_size - x_test.shape[1])
        Pic[pic_position:pic_position + x_test.shape[1], 
            pic_position:pic_position + x_test.shape[1]] = x_test[np.random.randint(0, x_test.shape[0]-1), :, :, 0]
        pred2 = model.predict([Pic.reshape(1, pic_size, pic_size, 1), mask])
        fig = pp.figure(figsize=(9, 9))
        fig.add_subplot(1, 2, 1)
        pp.matshow(pred2[0, :, :, 0], fignum=False)
        fig.add_subplot(1, 2, 2)
        pp.matshow(Pic, fignum=False)