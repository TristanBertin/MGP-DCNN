
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from MGP import MGP
import math
import h5py

train_x = np.arange(4)/4
test_x = np.arange(8)/4
train_y = np.array([[[1,2,3,4],[4,5,3,78],[7,8,5,12],[9,8,1,-9]], [[1,2,3,45],[4,5,4,32],[7,8,8,12],[9,8,7,45]]])
train_x = torch.tensor(train_x).float()
train_y = torch.tensor(train_y).float()
test_x = torch.tensor(test_x).float()
nb_tasks = 4

kernel = gpytorch.kernels.PeriodicKernel(period_length_prior=gpytorch.priors.NormalPrior(0.33,0.1))


n_iter = 20
learning_rate = 0.02


# for i in range(train_y.shape[0]):
#     mgp = MGP(kernel, learning_rate, n_iter, [[0], [1], [2]])
#     mgp.build_and_train_block_models(train_x, train_y[i])
#     # mgp.plot_model(train_x, train_y[i], test_x)



def train_MGP_multiple_individuals(x_train, y_train, block_indices, x_test=None, save_h5 = False, activate_plot=False):

    if save_h5 == True:
        list_means = []
        list_covariance_matrix = []

    for i in range(y_train.shape[0]):
        mgp = MGP(kernel, learning_rate, n_iter, block_indices)
        mgp.build_and_train_block_models(train_x, train_y[i])

        if activate_plot:
            mgp.plot_model(x_train, y_train[i], x_test)

        if save_h5:
            test_mean_list, test_covar_matrix_list, _ = mgp.test_block_model(x_test)

            list_means.append(test_mean_list)
            list_covariance_matrix.append(test_covar_matrix_list)



    if save_h5:

        h5_dataset = h5py.File('coucou')

        for i in range(len(block_indices)):
            cur_mean = np.array([list_means[i][0] for i in range(train_y.shape[0])])
            cur_covariance = np.array([list_covariance_matrix[i][0] for i in range(train_y.shape[0])])
            h5_dataset.create_dataset('mean_block_%d'%i, data=cur_mean)
            h5_dataset.create_dataset('covar_block_%d'%i, data=cur_covariance)
        h5_dataset.close()

        f1 = h5py.File('coucou', 'r')
        print(list(f1.keys()))




train_MGP_multiple_individuals(train_x, train_y, [[0],[1],[2,3]], test_x, True )



mean_popu, covar_popu = mgp.training_testing_mutliple_MGPs(train_x, train_y, test_x, plot=True)

h5_name = save_mean_covar_as_h5_file(mean_popu, covar_popu)
out = generate_posterior_samples(h5_name, 15)
print(h5_name)

plt.plot(out[0,2,:,0])
plt.plot(out[0,8,:,0])
plt.plot(out[0,6,:,0])
plt.show()