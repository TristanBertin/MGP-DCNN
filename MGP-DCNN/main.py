
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from MGP import Block_MGP
from skopt.space import Real, Integer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import data_processing
import math
import h5py
from TNN import Time_Neural_Network

nb_individuals = 60
nb_individuals_train = 40
nb_individuals_val = 10
nb_individuals_test = 10

nb_samples_per_id = 100
nb_time_steps = 105
nb_selected_points = 35
nb_peaks_selected = 2
nb_input_tasks = 5
nb_output_tasks = 5

block_indices = [[0],[1,2,3]]
nb_blocks = len(block_indices)


train_x = np.arange(4)/4
test_x = np.arange(8)/4
train_y = np.array([[[1,2,3,4],[4,5,3,12],[7,8,5,6],[9,8,1,-9]], [[1,2,3,5],[4,5,4,3],[7,8,8,12],[9,8,7,1]]])
train_x = torch.tensor(train_x).float()
train_y = torch.tensor(train_y).float()
test_x = torch.tensor(test_x).float()
test_y = torch.tensor(test_x).float()
val_y = torch.tensor(test_x).float()
val_x = torch.tensor(test_x).float()

kernel = gpytorch.kernels.PeriodicKernel(period_length_prior=gpytorch.priors.NormalPrior(0.33,0.1))
n_iter = 70
learning_rate = 0.1
data_augmentation_with_multiple_posteriors = True




def train_Block_MGP_multiple_individuals(x_train, y_train, block_indices, x_test=None, save_h5 = False, activate_plot=False):
    '''
    :param x_train: array size nb_timesteps_test * 1, represents time
    :param y_train: array size nb_individuals * nb_timesteps_test * number_tasks, represents time
    :param block_indices: list of lists of indices (ex: [[0,1],[2,3],[4]]
    :param x_test: array size nb_timesteps_test * 1, represents time
    :param save_h5: boolean, to save the test values in a h5 file or not
    :param activate_plot: to plot for each individual the resulted regressions, losses...
    :return: train Block MGP for multiple individuals

    BE CAREFUL : x_train and x_test must be the same for all the individuals...
    '''

    if len(x_train.shape)>1:
        raise ValueError('Wrong dimensions for the input X_train, x_train should be a 1D Vector')
    if len(x_test.shape)>1:
        raise ValueError('Wrong dimensions for the input X_test, x_test should be a 1D Vector')
    if x_train.shape[0] != y_train.shape[1]:
        raise ValueError('Number of time steps is different for x_train and y_train')


    flat_indices = [item for sublist in block_indices for item in sublist]
    nb_individuals, nb_time_steps, nb_tasks = y_train.shape

    if max(flat_indices) > nb_tasks:
        raise ValueError('One of the block indices is higher than the number of tasks in Y_train')

    if save_h5 == True:
        list_means = []
        list_covariance_matrix = []

    for i in range(nb_individuals):
        print('###########      INDIVIDUAL %d    ###########'%i)
        mgp = Block_MGP(kernel, learning_rate, n_iter, block_indices)
        mgp.build_and_train_block_models(train_x, train_y[i])

        if activate_plot:
            mgp.plot_model(x_train, y_train[i], x_test)

        if save_h5:
            test_mean_list, test_covar_matrix_list, _ = mgp.test_block_model(x_test)
            print('1111111111',len(test_mean_list))
            print('1111122222211111', len(test_covar_matrix_list))

            print('fdsdsdfsdfsdfsdfsdfsdfsdf', test_mean_list[0].shape)


            list_means.append(test_mean_list)
            list_covariance_matrix.append(test_covar_matrix_list)
            # print('RAAAAAA', test_mean_list.shape)
            # print('RAAAAAA-2222222222222', test_covar_matrix_list.shape)

    if save_h5:

        h5_dataset = h5py.File('output_models/OUTPUT_MGP_Nb_women_%d_Time_%d_selected_points_%d_Nb_blocks_%d_nb_peaks_%d'
                                  %(nb_individuals, nb_time_steps, nb_selected_points, len(block_indices), nb_peaks_selected), 'w')

        for i in range(len(block_indices)):
            cur_mean = np.array([list_means[i][0] for i in range(train_y.shape[0])])
            cur_covariance = np.array([list_covariance_matrix[i][0] for i in range(train_y.shape[0])])
            h5_dataset.create_dataset('mean_block_%d'%i, data=cur_mean)
            h5_dataset.create_dataset('covar_block_%d'%i, data=cur_covariance)
        h5_dataset.close()



train_Block_MGP_multiple_individuals(train_x, train_y, [[0],[1,2,3]], test_x, save_h5 = True, activate_plot=True)






if data_augmentation_with_multiple_posteriors: # if we draw multiple sample from the posterior distribution

    with h5py.File('output_models/OUTPUT_MGP_Nb_women_%d_Time_%d_selected_points_%d_Nb_blocks_%d_nb_peaks_%d'
                   # % (nb_individuals, nb_time_steps,  nb_selected_points, nb_blocks, nb_peaks_selected), 'r') as data:
                   % (2, 4, 35, 2, 2), 'r') as data:

        for i in range(nb_blocks):
            y_mean = data['mean_block_%d'%i][:]
            y_covar = data['covar_block_%d'%i][:]
            print(y_mean.shape)
            print(y_covar.shape)


        # y_data = data['y_data'][:]
        #
        # y_covar_1 = data['covar_1'][:]
        # y_covar_2 = data['covar_2'][:]
        # y_std = data['y_gp_out_std'][:]



x_train, y_train, x_val, y_val, x_test, y_test = data_processing.split_dataset_into_train_val_test(x_data, y_data, y_gp_posteriors=None)






'''  Scale the data  '''
scaler = StandardScaler()
shape1, shape2 = y_data.shape, y_gp_posterior_samples.shape

y_data = scaler.fit_transform(y_data.reshape(-1,5)).reshape(shape1)
y_gp_posterior_samples = scaler.transform(y_gp_posterior_samples.reshape(-1, 5)).reshape(shape2)


'''  Build the index to split the data  '''
new_index_women = np.random.permutation(np.arange(hyper_param['nb_women']))
index_women_train = new_index_women[:hyper_param['nb_women_train']]
index_women_val = new_index_women[hyper_param['nb_women_train'] : hyper_param['nb_women_train'] + hyper_param['nb_women_val']]
index_women_test = new_index_women[hyper_param['nb_women_train'] + hyper_param['nb_women_val'] : hyper_param['nb_women_train'] + hyper_param['nb_women_val'] + hyper_param['nb_women_test']]

'''  Split the data and Shuffle the training data   '''
shuffle_index_train = np.random.permutation(np.arange(hyper_param['nb_women_train'] * hyper_param['nb_samples_per_id']))

x_train = y_gp_posterior_samples[index_women_train].reshape(-1, hyper_param['nb_time_steps'], hyper_param['nb_hormones'])
x_train = x_train[shuffle_index_train]
y_train = np.tile(y_data[None, index_women_train], (hyper_param['nb_samples_per_id'],1,1,1))
y_train = np.swapaxes(y_train, 0,1).reshape(-1, hyper_param['nb_time_steps'], hyper_param['nb_hormones'])
y_train = y_train[shuffle_index_train]

x_val = y_gp_posterior_samples[index_women_val].reshape(-1, hyper_param['nb_time_steps'], hyper_param['nb_hormones'])
y_val = np.tile(y_data[None, index_women_val], (hyper_param['nb_samples_per_id'],1,1,1))
y_val = np.swapaxes(y_val, 0,1).reshape(-1, hyper_param['nb_time_steps'], hyper_param['nb_hormones'])

x_test = y_gp_posterior_samples[index_women_test].reshape(-1, hyper_param['nb_time_steps'], hyper_param['nb_hormones'])
y_test = np.tile(y_data[None, index_women_test], (hyper_param['nb_samples_per_id'],1,1,1))
y_test = np.swapaxes(y_test, 0,1).reshape(-1, hyper_param['nb_time_steps'], hyper_param['nb_hormones'])






dim_learning_rate = Real(low=2e-3, high=1.2e-2, prior='log-uniform', name='learning_rate')
dim_nb_layers = Integer(low=3, high=6, name='nb_layers')
dim_nb_filters = Integer(low=5, high=12, name='nb_filters')
dim_kernel_size = Integer(low=2, high=4, name='kernel_size')
dim_regularizer_coef = Real(low=1e-7, high=1e-3, prior='log-uniform', name='regularizer_coef')

parameters_range = [dim_learning_rate,dim_nb_layers,dim_nb_filters,dim_kernel_size,dim_regularizer_coef]

a = Time_Neural_Network('CNN', nb_time_steps=105, nb_tasks_input=5, nb_tasks_output=5, nb_selected_points=35,
                        nb_peaks_selected=0, batch_size=10, nb_epochs=10,
                        x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y, x_val=val_x, y_val=val_y)










mean_popu, covar_popu = mgp.training_testing_mutliple_MGPs(train_x, train_y, test_x, plot=True)

h5_name = save_mean_covar_as_h5_file(mean_popu, covar_popu)
out = generate_posterior_samples(h5_name, 15)
print(h5_name)

plt.plot(out[0,2,:,0])
plt.plot(out[0,8,:,0])
plt.plot(out[0,6,:,0])
plt.show()