import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import h5py
from sklearn.preprocessing import RobustScaler,MinMaxScaler, StandardScaler
import random

import sys
sys.path.insert(0, 'C:/Users/tmb2183/Desktop/myhmc/dev_tristan/Data_prediction/MGP/utils')
import data_processing

# L_selected_points = [16, 18, 20,22,24,26,28,30,32,34,35,36,38,40,42, 44]
# L_accuracy = [7.25,7.02, 6.72, 6.58, 6.02, 6.04, 6.0, 5.82, 5.79, 5.71, 5.49, 5.40, 5.41, 5.32, 5.30, 5.26]
# plt.plot(L_selected_points, L_accuracy)
# plt.title('test MSE')
# plt.show()

''' Multi output Gaussian Process, we try to predict the 5 levels of a single women with the trainig data of the same woman'''

# random.seed(42)
# np.random.seed(42) # cpu vars
# torch.manual_seed(42) # cpu  vars

'''################################            LOAD DATA         #########################################'''

data_file = 'C:/Users/tmb2183/Desktop/myhmc/data/dataset_N_50_Sub_1_T_200_freq_1_reselected'
config = 1
nb_cache = 12 # at the end it will nb_peak + nb_cache
peaks_selection = True



with h5py.File(data_file, 'r') as data:
    if config == 1:
        y_data = data['x_data'][:,0,:,0:2]
    if config == 2:
        y_data = data['x_data'][:,0,:,2:5]
    parameters = data['y_data'][:]

y_data = data_processing.align_data_on_peak(y_data)
shape = y_data.shape


# A = np.log(0.05 + (y_data[0,:,1]-np.min(y_data[0,:,1]))/np.max(y_data[0,:,1]))
# plt.plot(A)
# plt.show()
#
# plt.hist((np.log(y_data[0,:,0]/np.max(y_data[0,:,0]))+ 1))
# plt.show()




y_data = y_data[0]
total_time = shape[-2]
training_points = int(0.67 * total_time)

# fig,ax = plt.subplots(2,5)
# for i in range(5):
#     ax[0,i].hist(y_data[0,:,i],bins = 30)

if config==1:
    scaler = StandardScaler()
if config ==2:
    scaler = StandardScaler()

y_data = scaler.fit_transform(y_data)

# y_data[:,2:] = scaler.fit_transform(y_data[:,2:])
# y_data[:,0:2] = np.log(1 + y_data[:,0:2])
# maxi = np.max(y_data[:,0:2], axis=0)
# y_data[:,0:2] = y_data[:,0:2]/maxi
# print('ff', np.max(y_data[:,1]))
#

yy_train, y_test = y_data[:training_points], y_data[training_points:]
xx_train, x_test = np.arange(0,training_points)/training_points, np.arange(training_points, y_data.shape[-2])/training_points

# for i in range(5):
#     ax[1,i].hist(y_data[:,i], bins = 30)
# fig.suptitle('With Standard()', fontsize=16)
# plt.show()


if peaks_selection == True:
    idx_peaks = np.argsort(yy_train[:40,0])[-2:]
    sort_cache = np.random.permutation(np.delete(np.arange(yy_train.shape[0]), idx_peaks))

    filter = np.sort(np.concatenate([sort_cache[:nb_cache],idx_peaks]))
    false_filter = np.sort(sort_cache[nb_cache:])

else:
    nb_cache = nb_cache + 2
    sort_cache = np.random.permutation(np.arange(yy_train.shape[0]))
    filter = np.sort(sort_cache[:nb_cache])
    false_filter = np.sort(sort_cache[nb_cache:])


# print('three peaks', idx_peaks, idx_peaks.shape)
# print('filter', filter)
# print('ff filter', false_filter)




# plt.scatter(idx_sort_y[-3:], yy_train[:,0][idx_sort_y[-3:]])
# plt.show()







x_train, y_train = xx_train[filter], yy_train[filter]
single_x = np.arange(0,training_points)/training_points

xx, yy = xx_train[false_filter], yy_train[false_filter]

print('X TRAIN', x_train.shape, 'Y TRAIN', y_train.shape)
print('X TEST', x_test.shape, 'Y TEST', y_test.shape)



'''###############################       MODEL    #########################################'''

train_x = torch.tensor(x_train).float()
train_y = torch.tensor(y_train).float()
single_x = torch.tensor(single_x).float()

if config==1:
    class MultitaskGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)

            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ZeroMean(),num_tasks=2)

            self.covar_module = gpytorch.kernels.MultitaskKernel(
                    gpytorch.kernels.PeriodicKernel(
                        period_length_prior=gpytorch.priors.NormalPrior(0.33, 0.1)
                        # period_length_prior = gpytorch.priors.NormalPrior(-0.00065, 0.00001),
                        # lengthscale_prior = gpytorch.priors.NormalPrior(-0.06, 0.01)
                        ),
                num_tasks=2, rank=1)
            self.final_covariance_matrix = 0

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            go = gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
            self.final_covariance_matrix = go.covariance_matrix
            return go

        def return_covar_matrix(self,x):
            return gpytorch.distributions.MultitaskMultivariateNormal(self.mean_module(x), self.covar_module(x)).covariance_matrix

if config==2:
    class MultitaskGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)

            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ZeroMean(), num_tasks=3)

            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.PeriodicKernel(
                    period_length_prior=gpytorch.priors.NormalPrior(0.33, 0.1)
                    # period_length_prior = gpytorch.priors.NormalPrior(-0.00065, 0.00001),
                    # lengthscale_prior = gpytorch.priors.NormalPrior(-0.06, 0.01)
                ),
                num_tasks=3, rank=1)
            self.final_covariance_matrix = 0

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            go = gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
            self.final_covariance_matrix = go.covariance_matrix
            return go

        def return_covar_matrix(self, x):
            return gpytorch.distributions.MultitaskMultivariateNormal(self.mean_module(x),
                                                                      self.covar_module(x)).covariance_matrix





if config==1:
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
if config==2:
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
# likelihood.raw_noise.requires_grad = False
# likelihood.noise_covar.raw_noise.requires_grad = False


model = MultitaskGPModel(train_x, train_y, likelihood)
# print(model)



# Find optimal model hyperparameters
model.train()
likelihood.train()

if config==1:
    optimizer = torch.optim.Adam([{'params': model.parameters()},], lr=0.03)
    n_iter = 150

if config==2:
    optimizer = torch.optim.Adam([{'params': model.parameters()},], lr=0.02)
    n_iter = 170


# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)



# import GPy as gg
# kern_upper = gg.kern.Matern32(input_dim=1, variance=1.0, lengthscale=2.0, active_dims=[0], name='upper')
# kern_lower = gg.kern.Matern32(input_dim=1, variance=0.1, lengthscale=4.0, active_dims=[0], name='lower')
# k_hierarchy = gg.kern.Hierarchical(kernels=[kern_upper, kern_lower])
#
Loss =[]

for i in range(n_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
    optimizer.step()
    Loss.append(loss.item())

plt.plot(Loss)
plt.show()
#
# final_covar = model.final_covariance_matrix.detach().numpy()
# final_covar = np.around(final_covar, 4) #BECAUSE OF THE COMPUTATION, THE MATRIX IS NOT TOTALY SYMETRIC ! WE APPROXIMATE IT
# assert (final_covar.transpose() == final_covar).all()


# plt.imshow(final_covar.transpose()- final_covar)
# plt.colorbar()
# plt.show()
#
# plt.imshow(final_covar)
# plt.colorbar()
# plt.show()

# len_covar = final_covar.shape[0]
# new_index = np.array([np.arange(i,len_covar,2) for i in range(3)]).reshape(-1)
# new_index = np.array([[i,j] for i in new_index for j in new_index]).reshape(len_covar,len_covar,3)
# a = np.zeros((len_covar,len_covar))
# for i in range(len_covar):
#     for j in range(len_covar):
#         a[i,j] = final_covar[new_index[i,j,0], new_index[i,j,1]]
#
# plt.imshow(a)
# plt.colorbar()
# plt.show()

test_x = torch.tensor(x_test).float()

# print(model.return_covar_matrix(test_x))

# Set into eval mode
model.eval()
likelihood.eval()

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():

    test_observed_pred = likelihood(model(test_x))
    # Get mean
    test_mean = test_observed_pred.mean
    # Get lower and upper confidence bounds
    test_lower, test_upper = test_observed_pred.confidence_region()
    train_observed_pred = likelihood(model(single_x))
    # Get mean
    train_mean = train_observed_pred.mean
    # Get lower and upper confidence bounds
    train_lower, train_upper = train_observed_pred.confidence_region()

print('MAXXX', np.max(train_y.detach().numpy(), 0))





train_y = scaler.inverse_transform(train_y)
y_test = scaler.inverse_transform(y_test)
train_mean = scaler.inverse_transform(train_mean)
train_lower, train_upper = scaler.inverse_transform(train_lower), scaler.inverse_transform(train_upper)
test_lower, test_upper = scaler.inverse_transform(test_lower), scaler.inverse_transform(test_upper)
test_mean = scaler.inverse_transform(test_mean)
yy = scaler.inverse_transform(yy)

# plt.plot(test_mean[0])
# plt.plot(y_test[0])
# plt.show()

test_accuracy = np.average((test_mean - y_test)**2, axis=(0,1))/1000000
print('TEST', test_accuracy)

# train_y = train_y.detach().numpy()
# train_mean = train_mean.detach().numpy()
# train_lower, train_upper = train_lower.detach().numpy(), train_upper.detach().numpy()
# test_lower, test_upper = test_lower.detach().numpy(), test_upper.detach().numpy()
# test_mean = test_mean.detach().numpy()
#
# train_y[:,0:2] = f_inverse(train_y[:,0:2],maxi)
# train_mean[:,0:2] = f_inverse(train_mean[:,0:2],maxi)
# train_lower[:,0:2], train_upper[:,0:2] = f_inverse(train_lower[:,0:2],maxi), f_inverse(train_upper[:,0:2],maxi)
# test_lower[:,0:2], test_upper[:,0:2] = f_inverse(test_lower[:,0:2],maxi), f_inverse(test_upper[:,0:2],maxi)
# test_mean[:,0:2] = f_inverse(test_mean[:,0:2],maxi)

fig, ax = plt.subplots(5,1)

if config==1:
    for i in range(2):
        # Predictive mean as blue line
        ax[i].plot(single_x.numpy(), train_mean[:, i])
        ax[i].plot(test_x.numpy(), test_mean[:, i])

        ax[i].plot(train_x.detach().numpy(), train_y[:,i], 'k*', color = 'blue')
        ax[i].plot(xx, yy[:,i], 'k*', color='green')
        ax[i].plot(test_x.numpy(), y_test[:,i], 'k*', color = 'red')

        ax[i].fill_between(test_x.numpy(), test_lower[:, i], test_upper[:, i], alpha=0.5)
        ax[i].fill_between(single_x.numpy(), train_lower[:, i], train_upper[:, i], alpha=0.5)
        ax[i].legend(['Mean_train', 'Mean_test', 'Used training points', 'Unused training points', 'Test data', 'Test Confidence', 'Train Confidence'],
                     fontsize=7, loc='upper left')

if config==2:

    for i in range(3):
        # Predictive mean as blue line
        ax[i].plot(single_x.numpy(), train_mean[:, i])
        ax[i].plot(test_x.numpy(), test_mean[:, i])

        ax[i].plot(train_x.detach().numpy(), train_y[:, i], 'k*', color='blue')
        ax[i].plot(xx, yy[:, i], 'k*', color='green')
        ax[i].plot(test_x.numpy(), y_test[:, i], 'k*', color='red')

        ax[i].fill_between(test_x.numpy(), test_lower[:, i], test_upper[:, i], alpha=0.5)
        ax[i].fill_between(single_x.numpy(), train_lower[:, i], train_upper[:, i], alpha=0.5)
        ax[i].legend(
            ['Mean_train', 'Mean_test', 'Used training points', 'Unused training points', 'Test data', 'Test Confidence',
             'Train Confidence'],
            fontsize=7, loc='upper left')


for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

plt.show()

print('dd', xx.shape, yy.shape, x_train.shape)


# likelihood.raw_noise tensor([[-4.3097]])
# likelihood.noise_covar.raw_noise tensor([[-0.6085, -0.5299, -2.8402, -4.4719, -4.3749]])
# mean_module.base_means.0.constant tensor([[0.0999]])
# mean_module.base_means.1.constant tensor([[0.0982]])
# mean_module.base_means.2.constant tensor([[0.1823]])
# mean_module.base_means.3.constant tensor([[0.0289]])
# mean_module.base_means.4.constant tensor([[0.0195]])
# covar_module.task_covar_module.covar_factor tensor([[[-1.6568],
#          [-1.6157],
#          [-1.8798],
#          [ 0.0296],
#          [-0.5672]]])
# covar_module.task_covar_module.raw_var tensor([[-4.0760, -3.7822,  0.8085, -0.2957,  0.2559]])
# covar_module.data_covar_module.raw_lengthscale tensor([[[-2.9471]]])
# covar_module.data_covar_module.raw_period_length tensor([[[0.2898]]])

