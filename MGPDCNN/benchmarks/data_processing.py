
import numpy as np
import h5py


#FIXME; number of parameters ( = 2 ) has been hard coded

def file_name_to_param(file_name):
    params = file_name.split('_')
    params = [float(params[-1]),float(params[-3])]
    return params

def loading_data_single_file(data_path):
    '''
    :param data_path: path of the file that contains all the hormonal levels with a fixed set of parameters
    :return: an array of size [nb_samples, nb_sensors]
    '''

    file = open(data_path, "r")
    data = file.read().split("\n")
    data = [b.split(",") for b in data][:-1]
    data = np.array([[float(data[i][j]) for i in range(len(data))] for j in range(len(data[0]))])
    return data

def file_name_to_param(file_name):
    params = file_name.split('_')
    params = [float(params[-1]),float(params[-3])]
    return params

def split_data_single_file(data, training_ratio):

    nb_training_points = int(training_ratio * data.shape[0])

    data_train = data[:nb_training_points]
    data_test = data[nb_training_points::]

    return data_train, data_test


def generate_time_sequences(data, x_length, y_length, sampling_rate = 1 ):
    x = np.array([data[i:i+x_length:sampling_rate] for i in range(data.shape[0]-x_length-y_length)])
    y = np.array([data[i+x_length:i+x_length+y_length:sampling_rate] for i in range(data.shape[0]-x_length-y_length)])
    return x,y


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

    # x_data, y_data = data_preprocessing_whole_file('C:/Users/tmb2183/Desktop/myhmc/data/y_alpha_KmLH/')
    # h5_dataset = h5py.File('C:/Users/tmb2183/Desktop/myhmc/data/dataset_%d_t250' %x_data.shape[0], 'w')

    sampling_freq = 7
    time_length_seq = 33

    x_data, y_data = data_loader_variable_length('C:/Users/tmb2183/Desktop/myhmc/data/y_alpha_KmLH/', nb_data_subsets = 10, time_length_sequence = time_length_seq, sampling_frequency = sampling_freq)

    (nb_files, nb_data_subsets, nb_time_steps, nb_hormonal_levels) = x_data.shape

    h5_dataset = h5py.File('C:/Users/tmb2183/Desktop/myhmc/data/dataset_N_%d_Sub_%d_T_%d_freq_%d' %(nb_files, nb_data_subsets, time_length_seq, sampling_freq), 'w')
    h5_dataset.create_dataset('x_data', data=x_data)
    h5_dataset.create_dataset('y_data', data=y_data)
    h5_dataset.close()









