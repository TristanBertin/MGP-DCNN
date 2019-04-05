
from .MGP_subclasses import Multitask_GP_Model, Single_task_GP_model
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from .data_processing import change_representation_covariance_matrix
from scipy.signal import find_peaks
import h5py


class Block_MGP():

    def __init__(self, kernel, learning_rate, n_training_iter, block_indices):

        self.kernel = kernel
        self.learning_rate = learning_rate
        self.n_training_iter = n_training_iter

        self.block_indices = block_indices
        self.number_of_block = len(block_indices)
        self.total_nb_tasks = len([item for sublist in self.block_indices for item in sublist])

        self.model = []
        self.likelihood = []
        self.loss_list = []


    def build_and_train_single_model(self, x_train, y_train, block_number=0, smart_end = False):
        '''
        :param x_train: array size nb_timesteps *1, represents time
        :param y_train: array size nb_timesteps * nb_tasks
        :param block_number: the number of the block, starts from 0
        :return: modifies the attributes model and likelihood according to the training data
        '''


        nb_tasks = y_train.shape[-1]
        if nb_tasks == 1:
            self.likelihood.append(gpytorch.likelihoods.GaussianLikelihood())
            y_train = y_train[:,0]
            self.model.append(Single_task_GP_model(x_train, y_train, self.likelihood[block_number], self.kernel))

        if nb_tasks>1: #if no model has been ever trained, create a model
            self.likelihood.append(gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=nb_tasks))
            self.model.append(Multitask_GP_Model(x_train, y_train, self.likelihood[block_number], nb_tasks, self.kernel))

        self.model[block_number].train()
        self.likelihood[block_number].train()
        optimizer = torch.optim.Adam([{'params': self.model[block_number].parameters()}, ], lr=self.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood[block_number], self.model[block_number])
        loss_list_cur = []

        plot_frequency = self.n_training_iter // 5

        if smart_end:
            loss_hist = 0

        for i in range(self.n_training_iter):
            optimizer.zero_grad()
            output = self.model[block_number](x_train)
            loss = -mll(output, y_train)

            if i>30 and smart_end:
                min_loss_variation = np.min(np.array(loss_list_cur[1:30])-np.array(loss_list_cur[0:29]))
                if loss - loss_hist > - 2.5 * min_loss_variation :
                    break
                else:
                    loss.backward()
                    optimizer.step()
                    if i % plot_frequency == 0:
                        print('Iter %d/%d - Loss: %.3f' % (i + 1, self.n_training_iter, loss.item()))
                    loss_list_cur.append(loss.item())

            else:
                loss.backward()
                optimizer.step()
                if i % plot_frequency == 0:
                    print('Iter %d/%d - Loss: %.3f' % (i + 1, self.n_training_iter, loss.item()))
                loss_list_cur.append(loss.item())

            loss_hist = loss.item()

        self.loss_list.append(loss_list_cur)


    def build_and_train_block_models(self, x_train, y_train, smart_end = False):
        '''
        :param x_train: array size nb_timesteps *1, represents time
        :param y_train: array size nb_timesteps * nb_tasks
        :return: train the multiple MGP, one for each block
        '''
        for i in range(self.number_of_block):
            print('### BLOCK %d ###'%i)
            self.build_and_train_single_model(x_train, y_train[:,self.block_indices[i]], i, smart_end)


    def test_block_model(self, x_test):
        '''
        :param x_test: array size nb_timesteps_test * 1, represents time
        :return:  test_mean_list : the mean of the posterior MGPs
                  test_covar_matrix_list : the psoetrior covariance matrices
                  test_std :the standard deviation of the MGPs
        BE CAREFUL : the outputs are list, each block has then its own mean /coavriances arrays
        '''

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


    def plot_model(self, x_train, y_train, x_test, train_filter):
        '''
        :param x_train: array size nb_timesteps * 1, represents time
        :param y_train: array size nb_timesteps * nb_tasks
        :param x_test: array size nb_timesteps_test * 1, represents time
        :param train_filter : indices of the selected points for the training
        :return: a plot of the losses, the covariance matrices and the regression for each block
        '''

        test_mean_list, test_covar_matrix_list, test_std_deviation = self.test_block_model(x_test)

        fig = plt.figure(figsize=(18.5,9))
        gs = GridSpec(2, max(self.total_nb_tasks, 2*self.number_of_block))
        iter = 0

        for j in range(self.number_of_block):

            if len(self.block_indices[j])==1: #Single GP
                ax = fig.add_subplot(gs[0, iter])
                ax.plot(x_test.detach().numpy(), test_mean_list[j])
                ax.fill_between(x_test, test_mean_list[j] + test_std_deviation[j],
                                test_mean_list[j] - test_std_deviation[j], alpha=0.3)
                ax.set_title('Block %d Level %d' % (j, self.block_indices[j][0]))
                ax.plot(x_train.detach().numpy(), y_train.detach().numpy()[:, self.block_indices[j]],color='tomato')
                ax.plot(x_train.detach().numpy()[train_filter],y_train.detach().numpy()[train_filter, self.block_indices[j][0]], 'k*', color='red')
                iter = iter + 1
                ax.axvline(x_train.shape[0]/x_test.shape[0], color='green')

            else: #MGP
                for i in range(len(self.block_indices[j])):
                    ax = fig.add_subplot(gs[0, iter])
                    ax.plot(x_test.detach().numpy(), test_mean_list[j][:,i])
                    ax.fill_between(x_test, test_mean_list[j][:,i] + test_std_deviation[j][:,i], test_mean_list[j][:,i] - test_std_deviation[j][:,i], alpha=0.3)
                    ax.set_title('Block %d Level %d'%(j,self.block_indices[j][i]))
                    ax.plot(x_train.detach().numpy(), y_train.detach().numpy()[:, self.block_indices[j][i]], color='tomato')
                    ax.plot(x_train.detach().numpy()[train_filter], y_train.detach().numpy()[train_filter, self.block_indices[j][i]], 'k*', color='red')
                    ax.axvline(x_train.shape[0]/x_test.shape[0], color='green')
                    iter=iter+1


        for j in range(self.number_of_block):
            nb_tasks = len(self.block_indices[j])

            if nb_tasks ==1: #single GP
                ax1 = fig.add_subplot(gs[1, 2*j])
                ax1.imshow(test_covar_matrix_list[j])
                ax1.set_title('Block %d Covar Matrix' % j)
            if nb_tasks > 1 :  # multi GP
                ax1 = fig.add_subplot(gs[1, 2*j])
                matrix = change_representation_covariance_matrix(test_covar_matrix_list[j], nb_tasks)
                ax1.imshow(matrix)
                ax1.set_title('Block %d Covar Matrix' % j)

            ax2 = fig.add_subplot(gs[1, 2*j+1])
            ax2.plot(self.loss_list[j])
            ax2.set_title('Block %d Loss' % j)

        plt.show()




def train_Block_MGP_multiple_individuals(x_train, y_train, block_indices, x_test,
                                         kernel, learning_rate, n_iter,
                                         nb_selected_points = None, nb_peaks_selected = None,
                                         save_h5 = False, activate_plot=False, smart_end = False):
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

    flat_block_indices = [item for sublist in block_indices for item in sublist]

    a = []
    for i in range(len(block_indices)):
        a.append([])
        for j in range(len(block_indices[i])):
            a[i].append(flat_block_indices.index(block_indices[i][j]))
    block_indices = a


    if len(x_train.shape)>1:
        raise ValueError('Wrong dimensions for the input X_train, x_train should be a 1D Vector')
    if len(x_test.shape)>1:
        raise ValueError('Wrong dimensions for the input X_test, x_test should be a 1D Vector')
    if x_train.shape[0] != y_train.shape[1]:
        raise ValueError('Number of time steps is different for x_train and y_train')


    flat_indices = [item for sublist in block_indices for item in sublist]
    nb_individuals, _, nb_tasks = y_train.shape

    if max(flat_indices) > nb_tasks:
        raise ValueError('One of the block indices is higher than the number of tasks in Y_train')

    if save_h5 == True:
        list_means = []
        list_covariance_matrix = []

    for i in range(nb_individuals):

        if nb_selected_points != None: # WE SELECT A SUBSET OF POINTS

            if nb_peaks_selected != None:  # we select the peaks of the id_curve_peaks th curve

                nb_cache = nb_selected_points - nb_peaks_selected
                # idx_peaks = np.argsort(y_train[i, :40, id_curve_peaks])[-nb_peaks_selected:]

                ### scipy.find_peaks doesn' t detect the peak when it is the first element,
                # to avoid this problem, we add a single value at the begining...
                curve_peaks = np.concatenate((np.array([-10]), y_train[i, :40, 0]))
                idx_peaks,_ = find_peaks(curve_peaks, distance = 20)

                idx_peaks = idx_peaks - 1
                # print('LENGTH between selectec peaks :', idx_peaks[1]-idx_peaks[0])

                cache = np.random.permutation(np.delete(np.arange(x_train.shape[0]), idx_peaks[:nb_peaks_selected]))
                filter = np.sort(np.concatenate([cache[:nb_cache], idx_peaks]))

            if nb_peaks_selected == None:  # random selection over the training input timesteps
                nb_cache = nb_selected_points
                cache = np.random.permutation(np.arange(x_train.shape[0]))
                filter = np.sort(cache[:nb_cache])

            x_train_cur = x_train[filter]
            y_train_cur = y_train[i, filter]

        if nb_selected_points == None: # WE SELECT ALL THE POINTS

            x_train_cur = x_train
            y_train_cur = y_train[i]
            filter = np.arange(x_train.shape[0])


        print('###########      INDIVIDUAL %d    ###########'%i)
        mgp = Block_MGP(kernel, learning_rate, n_iter, block_indices)
        mgp.build_and_train_block_models(x_train_cur, y_train_cur, smart_end)

        if activate_plot:
            mgp.plot_model(x_train, y_train[i], x_test, train_filter = filter)

        if save_h5:
            test_mean_list, test_covar_matrix_list, _ = mgp.test_block_model(x_test)
            list_means.append(test_mean_list)
            list_covariance_matrix.append(test_covar_matrix_list)


    if save_h5:

        if nb_selected_points == None:
            nb_peaks_selected = x_train.shape[0]
        if nb_peaks_selected == None:
            nb_peaks_selected = 0

        h5_dataset_path = str('output_models/OUTPUT_MGP_Nb_individuals_%d_Time_%d_Selected_points_%d_Nb_blocks_%d_Nb_peaks_%d'
                                  %(nb_individuals, x_test.shape[0], nb_selected_points, len(block_indices), nb_peaks_selected))

        h5_dataset = h5py.File(h5_dataset_path, 'w')

        for i in range(len(block_indices)):
            cur_mean = np.array([list_means[j][i] for j in range(y_train.shape[0])])
            cur_covariance = np.array([list_covariance_matrix[j][i] for j in range(y_train.shape[0])])

            h5_dataset.create_dataset('mean_block_%d'%i, data=cur_mean)
            h5_dataset.create_dataset('covar_block_%d'%i, data=cur_covariance)
        h5_dataset.close()

        return h5_dataset_path






























