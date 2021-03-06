

from .TNN import Time_Neural_Network
from .MGP import train_Block_MGP_multiple_individuals
import .data_processing
from scipy.signal import resample, find_peaks

import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from skopt.space import Real, Integer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import math
import h5py
import numpy as np


###################################    IMPORT DATA    #####################################

data_file = 'C:/Users/tmb2183/Desktop/myhmc/data/dataset_N_60_Sub_1_T_150_freq_1'

with h5py.File(data_file, 'r') as data:
    y_data = data['x_data'][:]

y_data_shape = y_data.shape

nb_individuals = 60
nb_individuals_train = 40
nb_individuals_val = 10
nb_individuals_test = 10

nb_time_steps = 105
nb_train_time_steps = 70

nb_selected_points = 15
nb_peaks_selected = 2
nb_input_tasks = 5
nb_output_tasks = 5

nb_samples_per_id = 100

block_indices = [[0,1],[2,3,4]]
assert(np.concatenate(block_indices).shape[0] == nb_input_tasks)
nb_blocks = len(block_indices)

new_h5 = True
n_iter = 500
learning_rate_gp = 0.015
time_kernel = gpytorch.kernels.PeriodicKernel(period_length_prior = gpytorch.priors.NormalPrior(0.31,0.1))

ED_SAMPLING = True

###############            WE SCALE THE DATA BASED ON THE TRAINING SET      ##################

y_data = data_processing.align_data_on_peak(y_data, length=nb_time_steps, column=0)
y_data = y_data[:nb_individuals, :nb_time_steps]

sklearn_scaler = StandardScaler()

train_x, train_y, test_x, test_y, scaler = data_processing.prepare_data_before_GP(y_data,
                                                                  block_indices = block_indices,
                                                                  nb_time_steps = nb_time_steps,
                                                                  nb_train_time_steps = nb_train_time_steps,
                                                                  nb_train_individuals = nb_individuals_train,
                                                                  scaler=sklearn_scaler)

if new_h5 == True:
    h5_dataset_path = train_Block_MGP_multiple_individuals(train_x, train_y, block_indices, test_x,
                                                       kernel=time_kernel,
                                                       learning_rate=learning_rate_gp,
                                                       n_iter=n_iter,
                                                       nb_selected_points = nb_selected_points,
                                                       nb_peaks_selected = nb_peaks_selected,
                                                       activate_plot=False,
                                                       smart_end = True)

else :
    h5_dataset_path = 'output_GP/OUTPUT_MGP_Nb_individuals_%d_Time_%d_Selected_points_%d_Nb_blocks_%d_Nb_peaks_%d'\
                      %(nb_individuals, nb_time_steps, nb_selected_points, nb_blocks, nb_peaks_selected)

y_data = scaler.transform(y_data.reshape(-1, 5)).reshape(-1, 105, 5)


if ED_SAMPLING == True:

    h5_dataset_path = 'output_GP/ED_SAMPLING_OUTPUT_MGP_Nb_individuals_%d_Time_%d_Selected_points_%d_Nb_blocks_%d_Nb_peaks_%d'\
                      %(nb_individuals, nb_time_steps, nb_selected_points, nb_blocks, nb_peaks_selected)

    out_data = np.zeros(shape=(60,105,5))

    for i in range(y_data.shape[0]):
        idx_peaks, _ = find_peaks(y_data[i,:,0], distance=24)

        if idx_peaks[2] <72:
           idx_peak = idx_peaks[3]

        else:
            idx_peak = idx_peaks[2]

        data_cur = y_data[i, :idx_peak]

        for j in range(5):
            out_data[i,:,j] = resample(data_cur[:,j],105)
    y_data = out_data



x_train, y_train, x_val, y_val, x_test, y_test = \
    data_processing.import_and_split_data_train_val_test( output_gp_path = h5_dataset_path,
                                                          y_true = y_data,
                                                          block_indices= block_indices,
                                                          nb_timesteps = nb_time_steps,
                                                          nb_tasks = nb_input_tasks,
                                                          nb_individuals = nb_individuals,
                                                          nb_individuals_train = nb_individuals_train,
                                                          nb_individuals_val = nb_individuals_val,
                                                          nb_individuals_test = nb_individuals_test,
                                                          nb_samples_per_id=nb_samples_per_id,
                                                          plot_some_posteriors = False)


dim_learning_rate = Real(low=0.5e-3, high=2e-2, prior='log-uniform', name='learning_rate')
dim_nb_hidden_layers = Integer(low=3, high=6, name='nb_layers')
dim_nb_filters = Integer(low=5, high=12, name='nb_filters')
dim_regularizer_coef = Real(low=1e-8, high=1e-3, prior='log-uniform', name='regularizer_coef')
dim_kernel_size = Integer(low=2, high=9, name='kernel_size')
dim_dilation_factor = Integer(low=2, high=3, name='dilation_rate')
parameters_range = [dim_learning_rate,dim_nb_hidden_layers,dim_nb_filters,dim_regularizer_coef, dim_kernel_size, dim_dilation_factor]
default_parameters = [2e-3, 5, 8, 3e-8, 9, 2]


network = Time_Neural_Network('CNN', nb_selected_points=nb_selected_points, nb_peaks_selected=2, batch_size=50, nb_epochs=2,
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, x_val=x_val, y_val=y_val)


#network.build_TNN(learning_rate=0.002, nb_hidden_layers=5, nb_filters=8,regularizer_coef=1e-5, kernel_size=2, dilation_factor=2, display_summary=True)
# network.train_validate_TNN(regression_plot=True)

network.optimization_process(parameters_range, default_parameters=default_parameters, nb_calls = 200, nb_random_starts = 8, plot_conv=True)

