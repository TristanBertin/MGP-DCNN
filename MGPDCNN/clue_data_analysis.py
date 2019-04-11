import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from sklearn import mixture
import numpy as np




def plot_distributions(pd_data):

    dataset = pd_data['cycle_length']
    plt.hist(dataset, bins= 25)
    plt.show()

    dataset = pd_data['cycle_day_of_ovulation']
    plt.hist(dataset, bins= 100)
    plt.show()


def clean_dataset_by_gaussian_fit(pd_data, likelihood_boundary, basic_selection=False, plot_gaussian=False):
    '''

    :param pd_data: the dataset (Pandas)
    :param likelihood_boundary: the limit of the likelihood for the keeped points
    :param basic_selection: if True, we only elect the points by specifying for range for both features
    :param plot_gaussian: to plot Gaussian likelihood in case of basic_selection=True
    :return: the index of the selected points and the proportion of points that have been selected
    '''

    #1st Condition
    cycle_ovulation_data = pd_data[pd_data['cycle_length'] > pd_data['cycle_day_of_ovulation']] #logic...


    if basic_selection: # manually specify a range for both components
        selected_index = cycle_ovulation_data.index[(cycle_ovulation_data['cycle_length']>15) &
                                                    (cycle_ovulation_data['cycle_length']<40) &
                                                    (cycle_ovulation_data['cycle_day_of_ovulation']>5) &
                                                    (cycle_ovulation_data['cycle_day_of_ovulation']<30)].tolist()

    else:
        clf = mixture.GaussianMixture(n_components=1, covariance_type='full')

        clf.fit(cycle_ovulation_data)
        print('MEAN', clf.means_)
        print('COVAR', clf.covariances_)


        likelihood_data = -clf.score_samples(np.array([cycle_ovulation_data['cycle_length'], cycle_ovulation_data['cycle_day_of_ovulation']]).T)
        selected_index = np.where([likelihood_data < likelihood_boundary])[1]
        selected_proportion = selected_index.shape[0] / total_len
        print('Likelihood limit', likelihood_boundary, 'SELECTED PROPORTION', selected_proportion)

        if plot_gaussian:
            x = np.linspace(10, 90)
            y = np.linspace(0, 90)
            X, Y = np.meshgrid(x, y)
            XX = np.array([X.ravel(), Y.ravel()]).T
            Z = -clf.score_samples(XX)
            Z = Z.reshape(X.shape)
            CS = plt.contour(X, Y, Z, levels=100)
            CB = plt.colorbar(CS, shrink=0.8, extend='both')

            plt.hist2d(cycle_ovulation_data['cycle_length'], cycle_ovulation_data['cycle_day_of_ovulation'], bins=82)
            plt.scatter(cycle_ovulation_data['cycle_length'], cycle_ovulation_data['cycle_day_of_ovulation'], s=0.5,c='red')

            plt.title('Negative log-likelihood predicted by a GMM')
            plt.clabel(CS, inline=1, fontsize=10)
            plt.axis('tight')
            plt.xlabel('Cycle_length')
            plt.ylabel('Day of ovulation')
            plt.show()

    return selected_index, selected_proportion




def evolution_proportion_points():
    '''
    To look at the evolution of the keeped points depending on the likelihood boundary
    '''
    selected_proportion_list = []
    for limit in np.linspace(2,12,11):
        proportion,index = clean_dataset_by_gaussian_fit(cycle_ovulation_data, limit)[1], clean_dataset_by_gaussian_fit(cycle_ovulation_data, limit)[0]

        selected_proportion_list.append(proportion)


    # plt.plot(np.linspace(2.5,12,20), selected_proportion_list)
    # plt.xlabel('Likelihood limit')
    # plt.ylabel('Proportion of points')
    # plt.show()




if __name__ == '__main__':
    data = pd.read_pickle('data/ovulation_info.pickle')
    print("SHAPE OF INITIAL DATA", data.shape)
    cycle_ovulation_data = data[['cycle_length', 'cycle_day_of_ovulation']]

    selected_index = cycle_ovulation_data.index[(cycle_ovulation_data['cycle_length'] > 15) &
                                                (cycle_ovulation_data['cycle_length'] < 45) &
                                                (cycle_ovulation_data['cycle_day_of_ovulation'] > 0) &
                                                (cycle_ovulation_data['cycle_day_of_ovulation'] < 30)].tolist()

    print('tcheckk')

    print(cycle_ovulation_data.iloc[selected_index]['cycle_length'].var())
    print(cycle_ovulation_data.iloc[selected_index]['cycle_length'].mean())

    print(cycle_ovulation_data.iloc[selected_index]['cycle_day_of_ovulation'].var())
    print(cycle_ovulation_data.iloc[selected_index]['cycle_day_of_ovulation'].mean())

    plt.hist(cycle_ovulation_data.iloc[selected_index]['cycle_day_of_ovulation'])
    plt.show()


    cycle_ovulation_data = cycle_ovulation_data.iloc[selected_index]

    # cycle_ovulation_data= cycle_ovulation_data[cycle_ovulation_data['cycle_length'] < 40]
    # print(cycle_ovulation_data.shape)
    # cycle_ovulation_data = cycle_ovulation_data[cycle_ovulation_data['cycle_length'] < 40]
    total_len = cycle_ovulation_data.shape[0]


    # evolution_proportion_points()

    index, _ = clean_dataset_by_gaussian_fit(cycle_ovulation_data, plot_gaussian=False, likelihood_boundary=5.8)
    print(index.shape)
    cycle_ovulation_reselected = cycle_ovulation_data.iloc[list(index)]
    print('NEW SHAPE OF DATA', cycle_ovulation_reselected.shape)


    x = np.linspace(15, 45)
    y = np.linspace(0, 30)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    clf = mixture.GaussianMixture(n_components=1, covariance_type='full')
    clf.fit(cycle_ovulation_reselected)
    Z = -clf.score_samples(XX)
    Z = Z.reshape(X.shape)
    CS = plt.contour(X, Y, Z, levels=18, cmap='Oranges', linewidths=3)
    plt.hist2d(cycle_ovulation_reselected['cycle_length'], cycle_ovulation_reselected['cycle_day_of_ovulation'], bins=(29,29))
    plt.colorbar()
    plt.xlim((19,40))
    plt.xlabel('Cycle length', fontsize=34)
    plt.xticks(fontsize=24)
    plt.ylabel('Day of ovulation',fontsize=34)
    plt.yticks(fontsize=24)
    plt.ylim((6, 25))


    plt.show()













