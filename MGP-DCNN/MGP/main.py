
from .models import Multitask_GP_Model
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt



class MGP():

    def __init__(self, train_y_shape, likelihood, kernel, learning_rate, n_training_iter):
        self.nb_individuals, self.nb_time_steps, self.nb_tasks = train_y_shape
        self.likelihood = likelihood
        self.kernel = kernel
        self.learning_rate = learning_rate
        self.n_training_iter = n_training_iter


    def training_testing_MGP_single_id(self, train_x, train_y, test_x, plot_test):
        model = Multitask_GP_Model(train_x, train_y, self.likelihood, self.nb_tasks, self.kernel)
        model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam([{'params': model.parameters()}, ], lr=self.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, model)

        for i in range(self.n_training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            if i%10==0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, self.n_training_iter, loss.item()))

        model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():

            test_observed_pred = self.likelihood(model(test_x))
            test_mean = test_observed_pred.mean.detach().numpy()
            test_covar_matrix = model.return_covar_matrix(test_x).detach().numpy()


        if plot_test == True :

            test_lower, test_upper = test_observed_pred.confidence_region()
            test_lower, test_upper = test_lower.detach().numpy(), test_upper.detach().numpy()

            plt.plot(test_x.detach().numpy(), test_mean[:,0])
            plt.fill_between(test_x, test_upper[:,0], test_lower[:,0], alpha=0.3)
            plt.plot(train_x.detach().numpy(), train_y.detach().numpy()[:,0],'k*', color='red')
            plt.show()

        return test_mean, test_covar_matrix



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






















