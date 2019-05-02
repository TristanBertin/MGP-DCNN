
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import data_processing
from scipy.signal import resample, find_peaks
from MGP import train_Block_MGP_multiple_individuals
import gpytorch

data_file = 'C:/Users/tmb2183/Desktop/myhmc/data/according_Clue_dataset_N_60_Sub_1_T_150_freq_1'
sampling_list = [0, 35, 2, 69, 36, 6, 4, 8, 10, 12, 29, 67, 27, 23, 14, 19, 25, 52, 37, 33, 21, 55, 30, 59, 1, 68, 34, 38, 51, 3, 7, 40, 43, 47, 45, 28, 49]
time_kernel = gpytorch.kernels.PeriodicKernel(period_length_prior = gpytorch.priors.NormalPrior(0.31,0.1))

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




train_x, train_y, test_x, test_y, scaler = data_processing.prepare_data_before_GP(out_data,
                                                                  block_indices = [[0,1],[2,3,4]],
                                                                  nb_time_steps = 105,
                                                                  nb_train_time_steps = 70,
                                                                  nb_train_individuals = 40)

nb_selected_points = 15
print(train_x.shape)

h5_dataset_path = train_Block_MGP_multiple_individuals(train_x, train_y, [[0,1],[2,3,4]],test_x,
                                                       kernel=time_kernel,
                                                       learning_rate=0.02,
                                                       n_iter=500,
                                                       nb_selected_points=nb_selected_points,
                                                       nb_peaks_selected=2,
                                                       activate_plot=False,
                                                       smart_end=True,
                                                       sampling_order=sampling_list)
