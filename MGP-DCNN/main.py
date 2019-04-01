import numpy as np
from MGP import coucou

coucou(2)

def no_nan(function):

    def function_modified(*args, **kwargs):
        iter = 0
        print('######   FUNCTION : %s   ######'%function.__name__)

        while iter < 20:

            current_array = function(*args, **kwargs)

            if iter == 0:
                history_array = np.copy(current_array)
            else:
                index_nan = np.argwhere(np.isnan(history_array))
                history_array[index_nan] = current_array[index_nan]

            if np.isnan(history_array).any() == False:
                return history_array

            else:
                print('Restart', np.argwhere(np.isnan(history_array)).shape)
    #
    return function_modified



@no_nan
def bar(cou, gg, ee):
    print("lunch function")
    return np.nan
