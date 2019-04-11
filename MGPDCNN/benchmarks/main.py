import h5py
import os
from models import *
from data_processing import file_name_to_param, loading_data_single_file, split_data_single_file, generate_time_sequences
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.stats import kde


np.random.seed(101)

'''################################            LOAD DATA         #########################################'''

#FIXME : be careful : ratio between x and y given by stride ( for CNN)  !! EX stride = 2 ---> x_length/y_lenght = 2
sampling_frequency = 1
x_time_length = 35
y_time_length = 35


# assert nb_time_steps < 250, " some files are only containing 250 time steps"
files_name_list = [name for name in os.listdir('C:/Users/tmb2183/Desktop/myhmc/data/y_alpha_KmLH') if ('y_clark' in name and ('t500' in name and int(name[-3::]) >= 500))]

# list of all the set of parameters 9files in the same order as files_name_list )
parameters_list = np.array([file_name_to_param(files_name_list[i]) for i in range(len(files_name_list))])

print('Number of files', parameters_list.shape[0])

data = loading_data_single_file('C:/Users/tmb2183/Desktop/myhmc/data/y_alpha_KmLH/' + files_name_list[0])

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

train_data, test_data = split_data_single_file(data, training_ratio = 0.5)

nb_hormonal_levels = data.shape[-1]

x_train , y_train = generate_time_sequences(train_data, x_time_length, y_time_length, sampling_rate=sampling_frequency)
x_test , y_test = generate_time_sequences(test_data, x_time_length, y_time_length, sampling_rate=sampling_frequency)

'''########################################################################################################'''


#
#
# '''################################       PREPARE TRAIN / TEST SETS        ###############################'''
#
# training_ratio = 0.8
# nb_training_sequences = int((nb_files) * training_ratio)
#
# ''' HERE WE ARE SURE THAT THE TEST SET COME FROM RANDOM PARAMETERS'''
# permutation_train_test = np.random.permutation(np.arange(0, nb_files))
# id_train = permutation_train_test[:nb_training_sequences]
# id_test =  permutation_train_test[nb_training_sequences::]
#
# ''' NOW WE CAN SHUFFLE AFTER TRAIN / TEST SPLIT '''
# x_train, y_train = shuffle_data(x_data[id_train], y_data[id_train])
# x_test, y_test = shuffle_data(x_data[id_test], y_data[id_test])
#
# print('x_data', x_train.shape, 'y_data', y_test.shape)
#
# ''' #############################################################################################################'''
#



'''####################################        MODEL PARAMETERS     ##############################################'''

BATCH_SIZE = 40
EPOCHS = 300
dropout_frac = 0.0

MODEL = 'RNN' #########   -------->>>> CHOOSE RNN OR CNN #################

if MODEL == 'CNN':
    model = CNN_time_prediction(x_length = x_time_length, y_length =y_time_length , nb_sensors = nb_hormonal_levels, dropout_ratio = dropout_frac)
    print('#######################     CNN     #########################')

if MODEL == 'RNN':
    model = RNN_time_prediction(x_length = x_time_length, y_length =y_time_length, nb_sensors = nb_hormonal_levels, dropout_ratio = dropout_frac)
    print('#######################     RNN     #########################')
    # FIXME ; redo the model of the RNN

model.summary()
callbacks_list = [
    ModelCheckpoint(
        filepath="best_models/%s_best_model_Tx_%d_Ty_%d_freq_%d_drop_%.2f.h5"%(MODEL, x_time_length, y_time_length, sampling_frequency, dropout_frac),
        monitor='val_loss', save_best_only=True),
    EarlyStopping(monitor='acc', patience=60)]

'''###########################################################################################################'''



'''################################    TRAINING ---- EVALUATION  NETWORK  #######################################'''

TRAINING = True

print(y_train.shape)

if TRAINING == True :
    score = model.fit(x_train,y_train,
            batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks=callbacks_list,validation_split=0.2,verbose=1)


model.load_weights("best_models/%s_best_model_Tx_%d_Ty_%d_freq_%d_drop_%.2f.h5"%(MODEL, x_time_length, y_time_length, sampling_frequency, dropout_frac))
score = model.evaluate(x_test, y_test, verbose=0)

print('\nTest loss:', score[0])
print('Test accuracy:', score[1])

output = model.predict(x_test)
output_train = model.predict(x_train)
# output = scaler.inverse_transform(output)
# y_test = scaler.inverse_transform(y_test)

'''##########################################################################################################'''



'''############################################      PLOTS      ####################################################'''

if True :  #First plot, density for each of the parameters (x=true values, y=prediction)
    fig, axes = plt.subplots(ncols=1, nrows=5, figsize=(20, 10))

    for i in range(5):
        axes[i].set_title('Hormone %d'%i)
        axes[i].plot(np.arange(x_test.shape[1]), x_test[45, :, i])
        axes[i].plot(np.arange(x_test.shape[1], x_test.shape[1] + y_test.shape[1] ), y_test[45,:,i])
        axes[i].plot(np.arange(x_test.shape[1], x_test.shape[1] + y_test.shape[1] ), output[45,:,i])

    plt.show()

if True: #Second plot : MSE matrix  for each set of parameters
    MSE_matrix, list_KmLH, list_alpha = generate_MSE_matrix(y_test, output)

    plt.imshow(MSE_matrix.T, cmap='hot', interpolation='nearest', extent=[500,800,0.7,0.8], origin='lower',aspect='auto')

    plt.xlabel('KmLH')
    plt.ylabel('alpha')
    plt.title('MSE')
    plt.colorbar(boundaries=np.linspace(0,0.003,50))
    plt.show()

'''#####################################################################################################'''

