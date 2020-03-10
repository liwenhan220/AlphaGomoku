import keras
import keras.backend as K
from keras.models import Model, load_model
from keras.layers import *
from keras.regularizers import l2
from keras.optimizers import RMSprop
import numpy as np
from Gomoku_v8 import Gomoku

g = Gomoku()
blocks = 5
nn_size = 64
c = 0.01

def conv_layer(inp):
    conv1 = Conv2D(nn_size, (g.win,g.win), padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(inp)
    bn1 = BatchNormalization()(conv1)
    relu1 = Activation('relu')(bn1)

    return relu1

def res_layer(inp):
    shortcut = inp
    
    conv1 = Conv2D(nn_size, (g.win,g.win), padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(inp)
    bn1 = BatchNormalization()(conv1)
    relu1 = LeakyReLU()(bn1)

    conv2 = Conv2D(nn_size, (g.win,g.win), padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(relu1)
    bn2 = BatchNormalization()(conv2)
    # keras.layers.add
    shortcut = Conv2D(1, (g.win,g.win), padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(shortcut)
    shortcut = Conv2D(1, (g.win,g.win), padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(shortcut)
    x = add([bn2, shortcut])
    relu2 = LeakyReLU()(x)

    return relu2

def value_head(inp):
    conv1 = Conv2D(1, (1,1), padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(inp)
    bn1 = BatchNormalization()(conv1)
    relu1 = LeakyReLU()(bn1)

    flatten = Flatten()(relu1)
    dense1 = Dense(256, activation='relu', kernel_regularizer=l2(c), bias_regularizer=l2(c))(flatten)
    dense2 = Dense(1, activation='tanh', kernel_regularizer=l2(c), bias_regularizer=l2(c))(dense1)

    return dense2

def policy_head(inp, out):
    conv1 = Conv2D(2, (1,1), padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(inp)
    bn1 = BatchNormalization()(conv1)
    relu1 = LeakyReLU()(bn1)

    flatten = Flatten()(relu1)
    dense = Dense(out, activation='softmax', kernel_regularizer=l2(c), bias_regularizer=l2(c))(flatten)

    return dense

def create_model(shape, out):
    inputs = keras.Input(shape=shape)
    
    layer = conv_layer(inputs)
    for _ in range(blocks):
        layer = res_layer(layer)
        
    val_outputs = value_head(layer)
    pol_outputs = policy_head(layer,out)
    outputs = [pol_outputs, val_outputs]
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=RMSprop(lr=2e-3),metrics=['accuracy'])
    return model


