import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt


class Multitask_GP_Model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, gpytorch_kernel):
        super(Multitask_GP_Model, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(),num_tasks=num_tasks)
        self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch_kernel, num_tasks=num_tasks, rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    def return_covar_matrix(self, x):
        return gpytorch.distributions.MultitaskMultivariateNormal(self.mean_module(x), self.covar_module(x)).covariance_matrix



class Single_task_GP_model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, gpytorch_kernel):
        super(Single_task_GP_model, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def return_covar_matrix(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x)).covariance_matrix



def training_testing_mutliple_MGPs(self, train_x, train_y, test_x, plot=False):
    print('\n----- TRAINING MULTIPLE MGP ------\n')
    test_mean_array = np.empty(shape=(self.nb_individuals, self.nb_time_steps, self.nb_tasks))
    test_covar_array = np.empty(shape=(self.nb_individuals, self.nb_time_steps * self.nb_tasks, self.nb_time_steps * self.nb_tasks))

    for i in range(self.nb_individuals):
        print('Iindividual %d/%d'%(i+1, self.nb_individuals))
        test_mean, test_covar_matrix = self.training_testing_MGP_single_id(train_x, train_y[i], test_x, plot)
        test_mean_array[i] = test_mean
        test_covar_array[i] = test_covar_matrix

    return test_mean_array, test_covar_array






