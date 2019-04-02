import random

from TNN_subclasses import RNN_time_prediction, CNN_time_prediction
from keras.callbacks import ModelCheckpoint, EarlyStopping
import time

import skopt

from skopt.plots import plot_convergence
from skopt import gp_minimize
import matplotlib.pyplot as plt




class Time_Neural_Network():

    def __init__(self, model_type, nb_time_steps, nb_tasks_input, nb_tasks_output, nb_selected_points, nb_peaks_selected, batch_size, nb_epochs,
                        x_train, y_train, x_test, y_test, x_val, y_val):


        ''' ################       FIXED FOR THE OPTIMIZATION       #####################'''
        self.nb_time_steps = nb_time_steps
        self.nb_tasks_input = nb_tasks_input
        self.nb_tasks_output = nb_tasks_output
        self.model_type = model_type
        self.nb_selected_points = nb_selected_points
        self.nb_peaks_selected = nb_peaks_selected
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.params_history = []
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_val = x_val
        self.y_val = y_val

        ''' ################       VARIABLES OF THE OPTIMIZATION       #####################'''
        self.learning_rate = None
        self.nb_layers = None
        self.nb_filters = None
        self.regularizer_coef = None
        self.kernel_size = None
        self.dilation_factor = None


    def build_TNN(self, learning_rate, nb_layers, nb_filters,
                        regularizer_coef=1e-7, kernel_size=0, dilation_factor=1):

        self.learning_rate = learning_rate
        self.nb_layers = nb_layers
        self.nb_filters = nb_filters
        self.regularizer_coef = regularizer_coef
        self.kernel_size = kernel_size
        self.dilation_factor = dilation_factor

        '''
        :param learning_rate:
        :param nb_layers:
        :param nb_filters: BE CAREFUL : in the case of CNN : of nb of filters
                                        in the case of RNN : = hidden dimension = dimension of state between two units
        :param kernel_size:
        :param regularizer_coef:
        :return: BUILD network
        '''

        kernel_size = (kernel_size,)

        ###########################                   BUILD MODEL                     #############################

        if self.model == 'RNN':
            print('--------------------- RNN ----------------------')
            self.model = RNN_time_prediction(nb_time_steps = self.nb_time_steps,
                                        nb_tasks_input = self.nb_tasks_input,
                                        nb_tasks_output= self.nb_tasks_output,
                                        regularizer_coef = regularizer_coef,
                                        nb_layers = nb_layers,
                                        nb_hidden_dimension = nb_filters,
                                        learning_rate=learning_rate,)


        if self.model_type == 'CNN':
            print('--------------------- CNN ----------------------')
            if kernel_size == 0:
                raise ValueError('CNN model needs a kernel_size different from 0')

            self.model = CNN_time_prediction(xy_length = self.nb_time_steps,
                                        nb_tasks_input = self.nb_tasks_input,
                                        nb_tasks_output= self.nb_tasks_output,
                                        number_of_layers=nb_layers,
                                        learning_rate=learning_rate,
                                        number_of_filters=nb_filters,
                                        kernel_size=kernel_size,
                                        regularizer_coef=regularizer_coef,
                                        dilation_factor= dilation_factor)

        self.model.summary()



    def train_validate_TNN(self, plot=False):

        if self.learning_rate == None:
            assert ValueError('You have to build a network before training it ---> call obj.build_TNN() ')

        callbacks_list = [
            ModelCheckpoint(
                filepath="best_models/%s_nb_selected_points_%d_Txy_%d_nb_peaks_selected_%d.h5" %
                         (self.model_type, self.nb_selected_points,self.nb_time_steps, self.nb_peaks_selected),
                monitor='val_loss', save_best_only=True),
            EarlyStopping(monitor='acc', patience=15)]

        t1 = time.time()
        self.model.fit(self.x_train, self.y_train,
                       batch_size=self.batch_size,
                       epochs=self.nb_epochs,
                       callbacks=callbacks_list,
                       verbose=0,
                       validation_data=[self.x_val, self.y_val])

        self.training_time = time.time() - t1

        self.model.load_weights(
            "best_models/%s_nb_selected_points_%d_Txy_%d_nb_peaks_selected_%d.h5" %
            (self.model_type, self.nb_selected_points, self.nb_time_steps, self.nb_peaks_selected))

        val_loss = self.model.evaluate(self.x_val, self.y_val, verbose=0)[0]

        print('Time', self.training_time, 'Learning_rate', self.learning_rate, '|| nb_layers', self.nb_layers, '|| nb_filters',
              self.nb_filters, '|| kernel_size', self.kernel_size[0], '|| regularizer_coef', self.regularizer_coef, '|| VAL LOSS', val_loss)

        if plot == True:
            output_test = self.model.predict(self.x_val)
            output_train = self.model.predict(self.x_train)
            self.output_plots(self.x_train, self.y_train, output_train, self.x_val, self.y_val, output_test, val_loss)


        self.params_history.append([self.learning_rate, self.nb_layers, self.nb_filters, self.kernel_size, self.regularizer_coef, val_loss])

        return val_loss


    def build_and_train(self, learning_rate, nb_layers, nb_filters,
                        regularizer_coef=1e-7, kernel_size=0, dilation_factor=1):

        self.build_TNN(self, learning_rate, nb_layers, nb_filters, regularizer_coef, kernel_size, dilation_factor)
        val_loss = self.train_validate_TNN(self.x_train, self.y_train, self.x_val, self.y_val, plot=False)
        return val_loss



    def optimization_process(self, x_train, y_train, x_val, y_val, range_parameters, plot=False):


        default_parameters = [0.008, 4, 7, 2, 4e-7]


        # score = build_network(default_parameters) # Try default parameters
        search_result = gp_minimize(func=self.build_and_train,
                                    dimensions=range_parameters,
                                    acq_func='EI',  # Expected Improvement.
                                    n_calls=1000,
                                    n_random_starts=20,
                                    x0=default_parameters)

        if plot==True:
            plot_convergence(search_result)

        return search_result


    def output_plots(self, x_train, y_train, output_train, x_test, y_test, output_test, param_dict, score):
        # print(str(score)[:5])
        fig, axes = plt.subplots(ncols=5, nrows=5, figsize=(20, 10))
        for j in range(5):
            id = random.randint(0, param_dict['nb_women_test'] * param_dict['nb_samples_per_id'] - 1)
            # if j==3:
            #     id = 819
            # if j == 4:
            #     id = 91

            for i in range(5):
                axes[i, j].set_title('H_%d Id_%d' % (i, id))
                axes[i, j].plot(output_test[id, :, i], 'orange', label='prediction RNN')
                axes[i, j].plot(x_test[id, :, i], label='prediction GP')
                axes[i, j].plot(y_test[id, :, i], label='truth')
                axes[i, j].legend()
        fig.suptitle('Test')
        plt.savefig('plots/%s_LOSS_%d_TEST_nb_selected_points_%d_Txy_%d_nb_peaks_selected_%d_model_number_%d' %
                    (param_dict['MODEL'], score * 1000, param_dict['nb_selected_points'], param_dict['nb_time_steps'],
                     param_dict['nb_peaks_selected'], param_dict['model_number']))
        # plt.show()

        fig, axes = plt.subplots(ncols=5, nrows=5, figsize=(20, 10))
        for j in range(5):
            id = random.randint(0, param_dict['nb_women_train'] * param_dict['nb_samples_per_id'] - 1)
            if j == 3:
                id = 1950
            if j == 4:
                id = 91
            for i in range(5):
                axes[i, j].set_title('H_%d Id_%d' % (i, id))
                axes[i, j].plot(output_train[id, :, i], 'orange', label='prediction RNN')
                axes[i, j].plot(x_train[id, :, i], label='prediction GP')
                axes[i, j].plot(y_train[id, :, i], label='truth')
                axes[i, j].legend()
        fig.suptitle('Train')
        plt.savefig('plots/%s_LOSS_%d_TRAIN_nb_selected_points_%d_Txy_%d_nb_peaks_selected_%d_model_number_%d' %
                    (param_dict['MODEL'], score * 1000, param_dict['nb_selected_points'], param_dict['nb_time_steps'],
                     param_dict['nb_peaks_selected'], param_dict['model_number']))
        # plt.show()








