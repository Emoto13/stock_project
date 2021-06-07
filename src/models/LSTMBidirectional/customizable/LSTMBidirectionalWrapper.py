import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd

from .LSTMBidirectionalForecaster import LSTMBidirectionalForecaster
from .PreProcessor import PreProcessor


# Wrapper around 
class LSTMBidirectionalWrapper:
    def __init__(self, dataframe=None,
                 units=128, n_steps=7,
                 neurons=3, activation='relu',
                 epochs=35, batch_size=32, 
                 dropout=0.0, days_ahead=30,
                 clipnorm=0,
                 checkpoint="LSTM_checkpoint", load=False):
        self.dataframe = dataframe
        self.days_ahead = days_ahead
        self.n_steps = n_steps
        self.load = load
        self.checkpoint = checkpoint
        self.__extract_train_and_test_data(self.dataframe.y.values)

        # Load tf model
        if load: self.model = self.model.load_weights(checkpoint)
        else: self.model = LSTMBidirectionalForecaster(units=units, n_steps=n_steps, neurons=neurons,
                                          epochs=epochs, batch_size=batch_size, activation=activation,
                                          dropout=dropout,  clipnorm=clipnorm, checkpoint=checkpoint)
    
    # Get train and test sets
    # Convert Xs to correct shape
    def __extract_train_and_test_data(self, values):
        self.X, self.y = PreProcessor.prepare_data(values, self.n_steps)
        self.X_train, self.y_train, self.X_test, self.y_test = PreProcessor.split(self.X, self.y)
        
        self.X = self.__reshape_input(self.X)
        self.X_train = self.__reshape_input(self.X_train)
        self.X_test = self.__reshape_input(self.X_test)
        
    def __reshape_input(self, array):
        return array.reshape((array.shape[0], array.shape[1], 1))
        
    # Run train and test
    def train_and_test(self):
        self.model.train(self.X_train, self.y_train)
        # Use best model 
        self.model.load_weights(self.checkpoint)
        self.model.test(self.X_test, self.y_test)
        
    
    def predict_ahead(self, X=None, starting_date=None ,days_ahead=30):
        dates = []
        output = []
        temp_predictions = list(X)
        # Insert dummy element
        temp_predictions.insert(0, 0)
        
        for i in range(1, days_ahead):
            lstm_input = np.array(temp_predictions[1:], dtype='float64')
            lstm_input = lstm_input.reshape([1, self.n_steps, 1])
            yhat = self.model.predict(tf.convert_to_tensor(lstm_input, dtype='float64'))
            temp_predictions.append(yhat[0][0])
            temp_predictions = temp_predictions[1:]
            output.append(yhat[0].numpy()[0][0])
            dates.append(starting_date + np.timedelta64(i, 'D'))
            print("Predicted date:", dates[-1], output[-1])
        result_df = pd.DataFrame(dict(ds = pd.Series(dates),
                                    yhat = pd.Series(output))) 
        return result_df
    
    def predict(self, X=None):
        return self.model.predict(X)
    
    def run(self):
        if not self.load:
            self.train_and_test()
        predictions = self.predict_ahead(self.dataframe.y.values[-self.n_steps:], starting_date=self.dataframe.ds.values[-1], days_ahead=self.days_ahead)
        return predictions