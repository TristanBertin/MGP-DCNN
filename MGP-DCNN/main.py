
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from MGP import Block_MGP, train_Block_MGP_multiple_individuals
from skopt.space import Real, Integer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import data_processing
import math
import h5py
from TNN import Time_Neural_Network

nb_individuals = 4
nb_individuals_train = 2
nb_individuals_val = 1
nb_individuals_test = 1

nb_samples_per_id = 10
nb_time_steps = 105
nb_selected_points = 35
nb_peaks_selected = 2
nb_input_tasks = 5
nb_output_tasks = 5

block_indices = [[0,1],[2,3,4]]
nb_blocks = len(block_indices)

data_file = 'C:/Users/tmb2183/Desktop/myhmc/data/according_Clue_dataset_N_60_Sub_1_T_150_freq_1'

with h5py.File(data_file, 'r') as data:
    y_data = data['x_data'][:,:105]

x_train  = np.arange(70)/105
x_test = np.arange(105)/105
y_train = y_data[:4,:70]
y_test = y_data[:4,:105]

train_x = torch.tensor(x_train).float()
train_y = torch.tensor(y_train).float()
test_x = torch.tensor(x_test).float()
test_y = torch.tensor(y_test).float()


time_kernel = gpytorch.kernels.PeriodicKernel(period_length_prior = gpytorch.priors.NormalPrior(0.33,0.1))
n_iter = 10
learning_rate = 0.1
data_augmentation_with_multiple_posteriors = False


h5_dataset_path = train_Block_MGP_multiple_individuals(train_x, train_y, block_indices, test_x,
                                                       kernel=time_kernel, learning_rate=learning_rate, n_iter=n_iter,
                                                       nb_selected_points = nb_selected_points, nb_peaks_selected = nb_peaks_selected,
                                                       save_h5 = True, activate_plot=False)



gp_output_file = 'output_models/OUTPUT_MGP_Nb_women_%d_Time_%d_selected_points_%d_Nb_blocks_%d_nb_peaks_%d'%(2, 8, 35, 2, 2)
                     # % (nb_individuals, nb_time_steps,  nb_selected_points, nb_blocks, nb_peaks_selected), 'r') as data:


gp_output_file = 'output_models/OUTPUT_MGP_Nb_women_%d_Time_%d_selected_points_%d_Nb_blocks_%d_nb_peaks_%d'%(4, 105, 35, 2, 2)

# FIXME : differnet number of tasks input/output
# FIXME  : ALIGN ON PEAKS

print(nb_time_steps, nb_input_tasks)

x_train, y_train, x_val, y_val, x_test, y_test = \
    data_processing.import_and_split_data_train_val_test( output_gp_path = gp_output_file,
                                                          y_true = y_data,
                                                          block_indices= block_indices,
                                                          nb_timesteps = nb_time_steps,
                                                          nb_tasks = nb_input_tasks,
                                                          nb_individuals = nb_individuals,
                                                          nb_individuals_train = nb_individuals_train,
                                                          nb_individuals_val = nb_individuals_val,
                                                          nb_individuals_test = nb_individuals_test,
                                                          data_augmentation_with_multiple_posteriors=True,
                                                          nb_samples_per_id=10)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

plt.plot(x_test[0,:,3])
plt.plot(x_test[1,:,3])
plt.plot(x_test[9,:,3])
plt.plot(x_test[8,:,3])
plt.show()


# '''  Scale the data  '''
# scaler = StandardScaler()
# shape1, shape2 = y_data.shape, y_gp_posterior_samples.shape
#
# y_data = scaler.fit_transform(y_data.reshape(-1,5)).reshape(shape1)
# y_gp_posterior_samples = scaler.transform(y_gp_posterior_samples.reshape(-1, 5)).reshape(shape2)


dim_learning_rate = Real(low=2e-3, high=1.2e-2, prior='log-uniform', name='learning_rate')
dim_nb_hidden_layers = Integer(low=3, high=6, name='nb_layers')
dim_nb_filters = Integer(low=5, high=12, name='nb_filters')
dim_regularizer_coef = Real(low=1e-7, high=1e-3, prior='log-uniform', name='regularizer_coef')
dim_kernel_size = Integer(low=2, high=4, name='kernel_size')
dim_dilation_factor = Integer(low=2, high=4, name='kernel_size')


parameters_range = [dim_learning_rate,dim_nb_hidden_layers,dim_nb_filters,dim_regularizer_coef, dim_kernel_size, dim_dilation_factor]

network = Time_Neural_Network('CNN', nb_selected_points=35, nb_peaks_selected=0, batch_size=10, nb_epochs=50,
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, x_val=x_val, y_val=y_val)

network.build_TNN(learning_rate=0.1, nb_hidden_layers=2, nb_filters=4,regularizer_coef=1e-7, kernel_size=6, dilation_factor=1, display_summary=True)

network.train_validate_TNN(regression_plot=True)

default_parameters = [2e-3, 4, 7, 4e-7, 2, 4]

network.optimization_process(parameters_range, default_parameters=default_parameters, nb_calls = 4, nb_random_starts = 1, plot_conv=True)




mean_popu, covar_popu = mgp.training_testing_mutliple_MGPs(train_x, train_y, test_x, plot=True)

h5_name = save_mean_covar_as_h5_file(mean_popu, covar_popu)
out = generate_posterior_samples(h5_name, 15)
print(h5_name)

plt.plot(out[0,2,:,0])
plt.plot(out[0,8,:,0])
plt.plot(out[0,6,:,0])
plt.show()