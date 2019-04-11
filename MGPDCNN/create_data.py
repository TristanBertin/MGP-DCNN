# import keras
# from keras.models import Sequential
# from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import h5py
import re
import os
import matplotlib.pyplot as plt
import random


#FIXME; number of parameters (=2) has been hard coded

def file_name_to_param(file_name):
    params = file_name.split('_')
    params = [float(params[-1]),float(params[-3])]
    return params


def data_loader_variable_length(data_folder_path, nb_data_subsets, time_length_sequence, sampling_frequency=1):

    '''
    :param data_folder_path: the path of the folder that contains a file(or many) for each set of parameters
    :param nb_data_subsets: the number of sequences you want to create
    :param time_length_sequence: total length period of a sequence (it means how many days between the first sample and the last one)
    :param sampling_frequency : we pick 1/sampling_frequency points (it means if =1, we take all the samples, if 2 , we take one over two)
    :return: a .h5 file that contains all the data with x_data[Nb_files, Nb_data_subsets, Nb_time_steps, Nb_hormones]
            and y_data a matrix with the associated parameters [Nb_files, Nb-parameters]---> we generate many subset with
            random starting points

    IMPORTANT
              - We only use the y_clark files that contain 5 hormonal levels and 500 time steps and with KmLH>=500
              - In this function, we reduce the number of samples to time_length_sequence//sampling_frequency
    '''

    # assert nb_time_steps < 250, " some files are only containing 250 time steps"
    files_name = [name for name in os.listdir(data_folder_path) if ('y_clark' in name and ('t500' in name and int(name[-3::]) >= 500))]
    selected = np.random.permutation(np.arange(len(files_name)))[:60]
    files_name = [files_name[i] for i in selected]

    ''' The foolowing line find the number of samples from the time_length_sequence and the sampling_frequency'''
    nb_time_steps = time_length_sequence//sampling_frequency + int(time_length_sequence%sampling_frequency !=0)

    x_data = np.empty((len(files_name), nb_data_subsets, nb_time_steps, 5))
    y_data = np.empty((len(files_name), 2))

    for (i, file_name) in enumerate(files_name):

        file = open(data_folder_path + file_name, "r")
        x_file_data = file.read().split("\n")
        x_file_data = [b.split(",") for b in x_file_data][:-1]
        x_file_data = np.array([[float(x_file_data[i][j]) for i in range(len(x_file_data))] for j in range(len(x_file_data[0]))])

        x_data_with_subset = np.empty((nb_data_subsets, nb_time_steps, 5))

        for j in range(nb_data_subsets):
            start = random.randint(0, x_file_data.shape[0] - time_length_sequence)
            sub_data = x_file_data[start : start + time_length_sequence:sampling_frequency]
            x_data_with_subset[j] = sub_data

        x_data[i] = x_data_with_subset
        y_data[i]= file_name_to_param(file_name)
        print(i)


    x_data,y_data =  np.array(x_data), np.array(y_data)

    return x_data, y_data



def shuffle_data(x_data, y_data):
    '''
    :param x_data: array of shape [nb_files, nb_subset_data, nb_timesteps, nb_lvels]
    :param y_data: array of shape [ nb_files, nb_parameters]

    :return: x_data : array of shape [nb_files*nb_subset, nb_timesteps, nb_lvels]
             y_data : shape [ nb_files*nb_subset_data, nb_parameters]

    ----> flatten the two dimensions  (nb_files, nb_subset_data) and shuffle along this new dimension
    THIS SHOULD BE EXECUTED AFTER THE DATA SPLIT TRAINING/TEST, in order to have a test dataset with unseen parameters
    '''

    (nb_files, nb_data_subsets, nb_time_steps, nb_h_levels) = x_data.shape

    id_permutation = np.random.permutation(np.arange(nb_files * nb_data_subsets))

    x_data = x_data.reshape((-1, nb_time_steps, nb_h_levels))
    x_data = x_data[id_permutation]
    y_data = np.repeat(y_data, nb_data_subsets, axis=0)
    y_data = y_data[id_permutation]

    return x_data, y_data




if __name__ == '__main__':

    sampling_freq = 1
    time_length_seq = 150

    x_data, y_data = data_loader_variable_length('C:/Users/tmb2183/Desktop/myhmc/data/y_alpha_KmLH/', nb_data_subsets = 1, time_length_sequence = time_length_seq, sampling_frequency = sampling_freq)

    (nb_files, nb_data_subsets, nb_time_steps, nb_hormonal_levels) = x_data.shape

    h5_dataset = h5py.File('C:/Users/tmb2183/Desktop/myhmc/data/dataset_N_%d_Sub_%d_T_%d_freq_%d_reselected' %(nb_files, nb_data_subsets, time_length_seq, sampling_freq), 'w')
    h5_dataset.create_dataset('x_data', data=x_data)
    h5_dataset.create_dataset('y_data', data=y_data)
    h5_dataset.close()








