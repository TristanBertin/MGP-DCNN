from keras.models import Model
from keras.layers import Input, Dense

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling2D, Flatten, Input, Dropout, LSTM, TimeDistributed, Reshape
from keras.models import Model
import numpy


def CNN_time_prediction(x_length, y_length , nb_sensors, dropout_ratio = 0):

    input = Input(shape=(x_length, nb_sensors))

    ''' Here we apply Dropout even if we are in the TEST phase, because training is always set to TRUE'''
    x = Dropout(dropout_ratio)(input,training=True)

    x = Conv1D(nb_sensors * 2, 10, activation='sigmoid',padding='same',name='conv1', strides =1)(x)
    x = Conv1D(nb_sensors * 2, 10, activation='sigmoid', padding='same', name='conv3', strides=1)(x)
    x = Conv1D(nb_sensors, 10, activation='sigmoid', padding='same', name='conv4', strides=1)(x)

    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # x = Flatten(name='flatten')(x)
    # x = Dense(y_length, activation='sigmoid', name='fc12')(x)

    model = Model(input, x, name='time_cnn')

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'], lr=100)

    return model



def RNN_time_prediction(x_length, y_length , nb_sensors, dropout_ratio = 0):

    input = Input(shape=(x_length, nb_sensors))

    ''' Here we apply Dropout even if we are in the TEST phase, because training is always set to TRUE'''
    x = Dropout(dropout_ratio)(input,training=True)


    x = LSTM(5, return_sequences = True)(x)
    x = TimeDistributed(Dense(5))(x)
    x = Flatten()(x)
    # x = Dense(5 * y_length)(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # x = Flatten(name='flatten')(x)
    # x = Dense(y_length * 5)(x)
    x = Reshape((y_length,5))(x)



    model = Model(input, x, name='time_cnn')

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model


