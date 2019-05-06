
from keras.layers import Dense, Conv1D, MaxPooling2D, Flatten, Input, Dropout, LSTM, TimeDistributed, Reshape, Maximum, Conv1D, RepeatVector, Permute, multiply, GRU, Add, UpSampling1D, MaxPooling1D
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
import numpy as np
import h5py
import data_processing

data_file = 'C:/Users/tmb2183/Desktop/myhmc/data/according_Clue_dataset_N_60_Sub_1_T_150_freq_1'

with h5py.File(data_file, 'r') as data:
    y_data = data['x_data'][:]
    print(y_data.shape)


y_data = data_processing.align_data_on_peak(y_data, length=110, column=0)
shape = y_data.shape
sklearn_scaler = MinMaxScaler()

y_data = sklearn_scaler.fit_transform(y_data.reshape(-1,5)).reshape(shape)
y_train = y_data[:50]
y_test = y_data[50:60]

in_shape = 102

mask = np.random.choice([0, 1], size=(y_train.shape[0],in_shape), p=[2/3, 1/3])
mask = np.tile(mask[..., None], (1,1,5))
y_train_input = y_train[:,:in_shape] * mask

training = True

def autoencoder(nb_time_inputs):

    input = Input(shape=(nb_time_inputs,))

    x = Dense(30, activation='sigmoid')(input)

    x = Dense(nb_time_inputs, activation='sigmoid')(x)

    model = Model(input, x, name='autoencoder')

    ada_optimizer = Adam(lr=0.002)
    model.compile(loss='mean_squared_error', optimizer=ada_optimizer, metrics=['accuracy'])

    return model


def autoencoder_CNN(nb_time_inputs):

    input = Input(shape=(nb_time_inputs,5))

    x = Conv1D(1, (10,), activation='relu', padding='same')(input)
    x = MaxPooling1D(pool_size=2)(x)

    # decoder
    x = Conv1D(5, (10,), activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)


    model = Model(input, x, name='autoencoder')

    ada_optimizer = Adam(lr=0.002)
    model.compile(loss='mean_squared_error', optimizer=ada_optimizer, metrics=['accuracy'])

    return model



model = autoencoder_CNN(in_shape)
model.summary()

callbacks_list = [
    ModelCheckpoint(
        filepath="best_auto.h5",
        monitor='val_loss', save_best_only=True),
    EarlyStopping(monitor='acc', patience=100)]

if training:
    model.fit(y_train_input, y_train[:,:in_shape],
                   batch_size=10,
                   epochs=1000,
                   callbacks=callbacks_list,
                   verbose=1,
                   validation_split=0.2)

model.load_weights("best_auto.h5")

mask = np.random.choice([0, 1], size=(y_test.shape[0],in_shape), p=[2/3, 1/3])
mask = np.tile(mask[..., None], (1,1,5))
y_test_in = y_test[:,:in_shape] * mask

test_out = model.predict(y_test_in)

plt.plot(test_out[8,:,0])
plt.plot(y_test[8,:,0])
plt.plot(y_test_in[8,:,0],'o')
plt.show()

plt.plot(test_out[6,:,1])
plt.plot(y_test[6,:,1])
plt.plot(y_test_in[6,:,1],'o')
plt.show()

plt.plot(test_out[3,:,2])
plt.plot(y_test[3,:,2])
plt.plot(y_test_in[3,:,2],'o')
plt.show()


plt.plot(test_out[1,:,4])
plt.plot(y_test[1,:,4])
plt.show()





