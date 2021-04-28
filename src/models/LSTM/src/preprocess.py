import numpy as np
from numpy.lib.function_base import diff
from sklearn.model_selection import TimeSeriesSplit


def prepare_data(timeseries_data, n_steps):
    X = []
    for i in range(len(timeseries_data)):
        # Find the end of pattern
        end_ix = i + n_steps
        # Check if we are beyond the sequence
        if end_ix > len(timeseries_data) - 1:
            break
        # Gather input and output parts of the pattern
        seq_x = timeseries_data[i:end_ix]
        X.append(seq_x)
    return np.array(X)

def split_data(values, difference=1):
    """Splits time series data into train and test.

    Args:
        values (Iterable): [Data to split]
        difference (int): [Split size difference of X and Y]
    Returns:
        X_train, y_train, X_test, y_test
    """
    X = values[:-difference]
    y = values[difference:]

    tscv = TimeSeriesSplit()
    for train_index, test_index in tscv.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        return X_train, y_train, X_test, y_test

def make_arrays_equal_length(x,y):
    difference = abs(len(x) - len(y))
    if len(x) > len(y):
        x = x[difference:]
    elif len(x) < len(y):
        y = y[difference:]
    return x, y