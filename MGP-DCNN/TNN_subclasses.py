from keras.layers import Dense, Conv1D, MaxPooling2D, Flatten, Input, Dropout, LSTM, TimeDistributed, Reshape, Maximum, Conv1D, RepeatVector, Permute, multiply, GRU, Add
from keras.layers import Activation
from keras.models import Model
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import random
from skopt.utils import use_named_args
from skopt import gp_minimize
import time


def RNN_time_prediction(nb_time_steps, nb_tasks_input, nb_tasks_output, regularizer_coef, nb_layers, hidden_dimensions):

    input = Input(shape=(nb_time_steps, nb_tasks_input))

    for i in range(nb_layers):
        if i ==0:
            x = LSTM(hidden_dimensions, return_sequences=True)(input)
        else:
            x = LSTM(hidden_dimensions, return_sequences=True)(x)

    x = TimeDistributed(Dense(nb_tasks_output, kernel_regularizer=regularizer_coef))(x)

    model = Model(input, x, name='time_cnn')

    ada_optimizer = Adam(lr=0.002)
    model.compile(loss='mean_squared_error', optimizer=ada_optimizer, metrics=['accuracy'])

    return model


def CNN_time_prediction(xy_length, nb_sensors, number_of_layers, learning_rate, kernel_size, number_of_filters, regularizer_coef):

    regularizer = l2(regularizer_coef)

    input = Input(shape=(xy_length, nb_sensors))

    for i in range(number_of_layers-1):
        if i ==0 :
            x = Conv1D(number_of_filters, kernel_size, padding='same', name='conv_%d' % i, strides=1,
                       activation='relu', kernel_regularizer=regularizer, dilation_rate=1)(input)
        else:
            y = Conv1D(number_of_filters, kernel_size, padding='same',name='conv_%d'%i, strides =1, activation= 'relu', kernel_regularizer=regularizer, dilation_rate=2)(x)
            x = Add()([x,y])

    x = Conv1D(nb_sensors, kernel_size, padding='same', strides=1, kernel_regularizer=regularizer, dilation_rate=1)(x)

    model = Model(input, x, name='time_cnn')

    ada_optimizer = Adam(lr = learning_rate)
    model.compile(loss='mean_squared_error', optimizer=ada_optimizer, metrics=['accuracy'])

    return model