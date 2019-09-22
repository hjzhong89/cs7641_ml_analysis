import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.utils as utils

def load_data(filepath, delimiter=',', skip_header=1, target_column=-1, shuffle=False):
    '''
    Load data from a CSV file into test and training data
    :param filepath:
    :param delimiter:
    :param skip_header:
    :param target_column:
    :return:
    '''
    data = np.genfromtxt(filepath, delimiter=delimiter, skip_header=skip_header)

    if shuffle:
        data = utils.shuffle(data)

    X, y = data[:, :-1], data[:, -1]
    return train_test_split(X, y, train_size=.8, random_state=1)
