
from MGP_methods import Multitask_GP_Model, Single_task_GP_model
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class MGP():

    def __init__(self, kernel, learning_rate, n_training_iter, block_indices):

        self.kernel = kernel
        self.learning_rate = learning_rate
        self.n_training_iter = n_training_iter

        self.block_indices = block_indices
        self.number_of_block = len(block_indices)
        self.total_nb_tasks = len([item for sublist in self.block_indices for item in sublist])

        self.model = [None] * self.number_of_block
        self.likelihood = [None] * self.number_of_block
        self.loss_list = []



    def build_and_train_single_model(self, x_train, y_train, nb_tasks, block_number=0):

        if self.model[block_number] == None and nb_tasks == 1:
            self.likelihood[block_number] = gpytorch.likelihoods.GaussianLikelihood()
            y_train = y_train[:,0]
            self.model[block_number] = Single_task_GP_model(x_train, y_train, self.likelihood[block_number], self.kernel)

        if self.model[block_number] == None and nb_tasks>1: #if no model has been ever trained, create a model
            self.likelihood[block_number] = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=nb_tasks)
            self.model[block_number] = Multitask_GP_Model(x_train, y_train, self.likelihood[block_number], nb_tasks, self.kernel)

        self.model[block_number].train()
        self.likelihood[block_number].train()
        optimizer = torch.optim.Adam([{'params': self.model[block_number].parameters()}, ], lr=self.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood[block_number], self.model[block_number])
        loss_list_cur = []

        for i in range(self.n_training_iter):
            optimizer.zero_grad()
            output = self.model[block_number](x_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()
            if i % 30 == 0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, self.n_training_iter, loss.item()))
            loss_list_cur.append(loss.item())
        self.loss_list.append(loss_list_cur)


    def build_and_train_block_models(self, x_train, y_train):
        for i in range(self.number_of_block):
            self.build_and_train_single_model(x_train, y_train[:,self.block_indices[i]], len(self.block_indices[i]), i)


    def test_block_model(self, x_test):

        test_mean_list = []
        test_covar_matrix_list = []
        test_std = []

        for i in range(self.number_of_block):

            self.model[i].eval()
            self.likelihood[i].eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_observed_pred = self.likelihood[i](self.model[i](x_test))
                test_mean= test_observed_pred.mean.detach().numpy()
                test_covar_matrix = self.model[i].return_covar_matrix(x_test).detach().numpy()

            test_mean_list.append(test_mean)
            test_covar_matrix_list.append(test_covar_matrix)

            test_lower, test_upper = test_observed_pred.confidence_region()
            test_lower, test_upper = test_lower.detach().numpy(), test_upper.detach().numpy()
            test_std.append((test_upper - test_lower) / 4)

        return test_mean_list, test_covar_matrix_list, test_std


    def plot_model(self, x_train, y_train, x_test):

        test_mean_list, test_covar_matrix_list, test_std_deviation = self.test_block_model(x_test)

        fig = plt.figure()
        gs = GridSpec(2, max(self.total_nb_tasks, 2*self.number_of_block))
        iter = 0

        for j in range(self.number_of_block):

            if len(self.block_indices[j])==1: #Single GP
                ax = fig.add_subplot(gs[0, iter])
                ax.plot(x_test.detach().numpy(), test_mean_list[j])
                ax.fill_between(x_test, test_mean_list[j] + test_std_deviation[j],
                                test_mean_list[j] - test_std_deviation[j], alpha=0.3)
                ax.set_title('Block %d Level %d' % (j, self.block_indices[j][0]))
                ax.plot(x_train.detach().numpy(), y_train.detach().numpy()[:, self.block_indices[j]], 'k*',color='tomato')
                iter = iter + 1

            else: #MGP
                for i in range(len(self.block_indices[j])):
                    ax = fig.add_subplot(gs[0, iter])
                    ax.plot(x_test.detach().numpy(), test_mean_list[j][:,i])
                    ax.fill_between(x_test, test_mean_list[j][:,i] + test_std_deviation[j][:,i], test_mean_list[j][:,i] - test_std_deviation[j][:,i], alpha=0.3)
                    ax.set_title('Block %d Level %d'%(j,self.block_indices[j][i]))
                    ax.plot(x_train.detach().numpy(), y_train.detach().numpy()[:, self.block_indices[j][i]], 'k*', color='tomato')
                    iter=iter+1


        for j in range(self.number_of_block):
            ax1 = fig.add_subplot(gs[1, 2*j])
            ax1.imshow(test_covar_matrix_list[j])
            ax1.set_title('Block %d Covar Matrix' % j)

            ax2 = fig.add_subplot(gs[1, 2*j+1])
            ax2.plot(self.loss_list[j])
            ax2.set_title('Block %d Loss' % j)

        plt.show()




























