
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import MGP


train_x = np.arange(4)/4
test_x = np.arange(8)/4
train_y = np.array([[[1,2],[4,5],[7,8],[9,8]], [[1,2],[4,5],[7,8],[9,8]]])
train_x = torch.tensor(train_x).float()
train_y = torch.tensor(train_y).float()
test_x = torch.tensor(test_x).float()
nb_tasks = 2


kernel = gpytorch.kernels.PeriodicKernel(period_length_prior=gpytorch.priors.NormalPrior(0.33,0.1))
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=nb_tasks)
n_iter = 30
learning_rate = 0.02


model = MGP.Multitask_GP_Model((2,8,2), likelihood, kernel, learning_rate, n_iter)

mean_popu, covar_popu = model.training_testing_mutliple_MGPs(train_x, train_y, test_x, plot=True)
h5_name = save_mean_covar_as_h5_file(mean_popu, covar_popu)
out = generate_posterior_samples(h5_name, 15)
print(h5_name)

plt.plot(out[0,2,:,0])
plt.plot(out[0,8,:,0])
plt.plot(out[0,6,:,0])
plt.show()