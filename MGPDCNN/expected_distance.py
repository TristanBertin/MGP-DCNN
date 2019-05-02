
import numpy as np
from scipy.signal import resample, find_peaks
import matplotlib.pyplot as plt
import h5py
import data_processing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import gpytorch
from scipy.stats import norm
import h5py

coef_smart = 1

sampling_list = [0, 35, 37, 3, 68, 42, 65, 14, 45, 50, 34, 69, 33, 19, 24, 62, 57, 1, 32, 47, 2, 40, 38, 52, 36, 39, 30, 41, 46, 67, 31, 53, 48, 55, 8, 26, 23]
sampling_list_2 = [0, 35, 2, 69, 36, 6, 4, 8, 10, 12, 29, 67, 27, 23, 14, 19, 25, 52, 37, 33, 21, 55, 30, 59, 1, 68, 34, 38, 51, 3, 7, 40, 43, 47, 45, 28, 49]

[0, 35, 2, 36, 34, 6, 10, 4, 14, 33, 1, 27, 23, 37, 54]


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(),num_tasks=num_tasks)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.PeriodicKernel(period_length_prior=gpytorch.priors.NormalPrior(0.33,0.1)),num_tasks=num_tasks, rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    def return_covar_matrix(self, x):
        return gpytorch.distributions.MultitaskMultivariateNormal(self.mean_module(x), self.covar_module(x)).covariance_matrix



def Multi_task_GP_single_id_double_GP(x_train,y_train, nb_total_time_prediction, lr_1, lr_2, nb_iter_1, nb_iter_2):

    train_x = torch.tensor(x_train).float()
    train_y_1 = torch.tensor(y_train[:,:2]).float()
    train_y_2 = torch.tensor(y_train[:,2:5]).float()

    likelihood1 = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model1 = MultitaskGPModel(train_x, train_y_1, likelihood1, num_tasks = 2)

    likelihood2 = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
    model2 = MultitaskGPModel(train_x, train_y_2, likelihood2, num_tasks=3)

    model1.train()
    likelihood1.train()
    model2.train()
    likelihood2.train()

    optimizer1 = torch.optim.Adam([{'params': model1.parameters()}, ], lr=lr_1)
    mll1 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood1, model1)
    optimizer2 = torch.optim.Adam([{'params': model2.parameters()}, ], lr=lr_2)
    mll2 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood2, model2)

    loss_hist = 0
    loss_list_cur1 = []

    print("BLOCK 1 ")
    for i in range(nb_iter_1):
        optimizer1.zero_grad()
        output1 = model1(train_x)
        loss1 = -mll1(output1, train_y_1)

        if i > 50:
            min_loss_variation = np.min(np.array(loss_list_cur1[1:30]) - np.array(loss_list_cur1[0:29]))
            if loss1 - loss_hist > - coef_smart * min_loss_variation:
                break
            else:
                loss1.backward()
                optimizer1.step()
                if i % 50 == 0:
                    print('Iter %d/%d - Loss: %.3f' % (i + 1, 500, loss1.item()))

        else:
            loss1.backward()
            optimizer1.step()
            if i % 50 == 0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, 500, loss1.item()))

        loss_list_cur1.append(loss1.item())
        loss_hist = loss1.item()




    loss_hist = 0
    loss_list_cur2 = []
    print("BLOCK 2 ")
    for i in range(nb_iter_2):
        optimizer2.zero_grad()
        output2 = model2(train_x)
        loss2 = -mll2(output2, train_y_2)

        if i > 50:
            min_loss_variation = np.min(np.array(loss_list_cur2[1:30]) - np.array(loss_list_cur2[0:29]))
            if loss2 - loss_hist > -coef_smart * min_loss_variation:
                break
            else:
                loss2.backward()
                optimizer2.step()
                if i % 50 == 0:
                    print('Iter %d/%d - Loss: %.3f' % (i + 1, 500, loss2.item()))

        else:
            loss2.backward()
            optimizer2.step()
            if i % 50 == 0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, 500, loss2.item()))

        loss_hist = loss2.item()
        loss_list_cur2.append(loss2)

    model1.eval()
    likelihood1.eval()
    model2.eval()
    likelihood2.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        xx_range = torch.tensor(np.arange(nb_total_time_prediction) / nb_total_time_prediction).float()
        pred_1 = likelihood1(model1(xx_range))
        pred_2 = likelihood2(model2(xx_range))

        mean_1 = pred_1.mean.detach().numpy()
        mean_2 = pred_2.mean.detach().numpy()

        low_1, up_1 = pred_1.confidence_region()
        low_2, up_2 = pred_2.confidence_region()

        std_1, std_2 = (up_1 - low_1).detach().numpy(), (up_2 - low_2).detach().numpy()

        covar_matrix = [model1.return_covar_matrix(xx_range).detach().numpy(), model2.return_covar_matrix(xx_range).detach().numpy()]
        mean = np.concatenate([mean_1, mean_2], axis=-1)
        std = np.concatenate([std_1, std_2], axis=-1) / (2 * 1.96) # because it's a 95% confidence intervall

        del model1
        del model2

    return mean, std, covar_matrix



def expected_distance(remaining_points, gp_mean, gp_std, y_full, list_selected_points, plot=False):
    ''' own acquisition function --> return the index of the next point to sample'''

    term1 = (gp_mean - y_full)
    term2 = 1 - 2*norm.cdf((y_full - gp_mean)/gp_std)
    term3 = 2*gp_std*norm.pdf((y_full - gp_mean)/gp_std)
    result_list = term1 * term2 + term3

    scaler = MinMaxScaler()
    result_list = scaler.fit_transform(result_list)
    # result_list[:,:2] = 1.2 * result_list[:,:2]
    result_list = np.sum(result_list,axis=-1)
    # result_list = result_list[:,0]

    remaining_points = np.array(remaining_points)
    restricted_list = result_list[remaining_points]

    index = np.argmax(restricted_list)

    return remaining_points[index]



data_file = 'C:/Users/tmb2183/Desktop/myhmc/data/according_Clue_dataset_N_60_Sub_1_T_150_freq_1'

with h5py.File(data_file, 'r') as data:
    y_data = data['x_data'][:]

scaler = StandardScaler()

y_data = data_processing.align_data_on_peak(y_data, length=110, column=0)
y_data = scaler.fit_transform(y_data.reshape(-1,5)).reshape(-1,110,5)

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

    # plt.plot(y_data[i, :70, 0])
    # plt.plot(out_data[i,:71,0])
    # # plt.axvline(x=idx_peak)
    # plt.show()

id = 44


list_selected_points = [0,35]
x_train = np.arange(70)/105
y_train = out_data[id,:70]
x_full = np.arange(105)/105
y_full = out_data[id]



nb_iter_1 = 50

for point_i in range(35):

    print("-------------  NEW POINT %d----------------------"%point_i)

    x_train_selected = x_train[list_selected_points]
    y_train_selected = y_train[list_selected_points]

    # FIXME ; replace with w_selected
    gp_subset_mean, gp_subset_std, subset_covariance_matrix = Multi_task_GP_single_id_double_GP(x_train_selected, y_train_selected,
                                                                                                nb_total_time_prediction=105,
                                                                                                nb_iter_1=280,
                                                                                                nb_iter_2=300,
                                                                                                # nb_iter_1=100,
                                                                                                # nb_iter_2=100,
                                                                                                lr_1=0.02,
                                                                                                lr_2=0.02)
    # plt.plot(x_full, gp_subset_mean[:, 0])
    # plt.fill_between(x_full, gp_subset_std[:,0]+ gp_subset_mean[:,0],-gp_subset_std[:,0]+ gp_subset_mean[:,0], alpha = 0.3)
    # plt.plot(x_full, y_full[:,0])
    # plt.scatter(x_train_selected, y_train_selected[:,0])
    # plt.savefig('step_by_step_acquisition_plots/nb_points_%d' %int(point_i+2))
    # plt.show()

    # plt.clf()
    # plt.plot(x_full, gp_subset_mean[:, 2])
    # plt.fill_between(x_full, gp_subset_std[:,2]+ gp_subset_mean[:,2],-gp_subset_std[:,2]+ gp_subset_mean[:,2], alpha = 0.3)
    # plt.plot(x_full, y_full[:,2])
    # plt.scatter(x_train_selected, y_train_selected[:,2])
    # plt.savefig('step_by_step_acquisition_plots/nb_points_%d' %int(point_i+2))
    # plt.show()


    remaining_points = list(np.setdiff1d(range(70), list_selected_points))

    index_best_expected_distance = expected_distance(remaining_points, gp_subset_mean, gp_subset_std, y_full, list_selected_points, plot=True)
    print(index_best_expected_distance)

    list_selected_points.append(index_best_expected_distance)
    print(list_selected_points)


print(list_selected_points)




def expected_distance(remaining_points, gp_mean, gp_std, y_full, list_selected_points, plot=False):
    ''' own acquisition function --> return the index of the next point to sample'''

    term1 = (gp_mean - y_full)
    term2 = 1 - 2*norm.cdf((y_full - gp_mean)/gp_std)
    term3 = 2*gp_std*norm.pdf((y_full - gp_mean)/gp_std)
    result_list = term1 * term2 + term3

    scaler = MinMaxScaler()
    result_list = scaler.fit_transform(result_list)
    # result_list[:,:2] = 1.2 * result_list[:,:2]
    result_list = np.sum(result_list,axis=-1)
    # result_list = result_list[:,0]

    remaining_points = np.array(remaining_points)
    restricted_list = result_list[remaining_points]

    index = np.argmax(restricted_list)




    return remaining_points[index]