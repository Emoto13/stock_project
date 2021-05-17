import numpy as np

class PreProcessor:

    @staticmethod
    def prepare_data(timeseries_data, n_steps):
        print(len(timeseries_data))
        X, y = [], []
        for i in range(len(timeseries_data)):
            # Find the end of pattern
            end_ix = i + n_steps
            # Check if we are beyond the sequence
            if end_ix > len(timeseries_data)-1:
            	break
            # Gather input and output parts of the pattern
            seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    
    @staticmethod
    def split(X, y, train_test_split=0.2):
        slice_index_X = int((1-train_test_split)*len(X))
        slice_index_y = int((1-train_test_split)*len(y))
        X_train, X_test = X[:slice_index_X], X[slice_index_X:]
        y_train, y_test = y[:slice_index_y], y[slice_index_y:]
        return X_train, y_train, X_test, y_test
