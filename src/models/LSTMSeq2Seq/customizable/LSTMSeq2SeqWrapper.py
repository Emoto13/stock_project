from .LSTMSeq2SeqForecaster import LSTMSeq2SeqForecaster
from .PreProcessor import PreProcessor

import pandas as pd
import numpy as np
import tensorflow as tf

# Wrapper around 
class LSTMSeq2SeqWrapper:
    def __init__(self, dataframe=None,
                 units=128, n_steps=7, n_steps_ahead=3, activation='relu',
                 epochs=35, batch_size=32, dropout=0.0,
                 days_ahead=30,
                 checkpoint="LSTMSeq2Seq_checkpoint.ckpt", load=False):
        self.dataframe = dataframe
        self.days_ahead = days_ahead

        self.n_steps = n_steps
        self.n_steps_ahead = n_steps_ahead

        self.load = load
        self.checkpoint = checkpoint
        self.__extract_train_and_test_data(self.dataframe.y.values)

        # Load tf model
        if load: self.model = self.model.load_weights(checkpoint)
        else: self.model = LSTMSeq2SeqForecaster(units=units, n_steps=n_steps, n_steps_ahead=n_steps_ahead, activation=activation, dropout=dropout, epochs=epochs, batch_size=batch_size, checkpoint=checkpoint)
    
    # Get train and test sets
    # Convert Xs to correct shape
    def __extract_train_and_test_data(self, values):
        self.X, self.y = PreProcessor.prepare_data(values, self.n_steps, self.n_steps_ahead)
        self.X_train, self.y_train, self.X_test, self.y_test = PreProcessor.split(self.X, self.y)
        
        self.X = self.__reshape_input(self.X)
        self.X_train = self.__reshape_input(self.X_train)
        self.X_test = self.__reshape_input(self.X_test)
        
    def __reshape_input(self, array):
        return array.reshape((array.shape[0], array.shape[1], 1))
        
    # Run train and test
    def train_and_test(self):
        self.model.train(self.X, self.y)
        # Use best model 
        self.model.load_weights(self.checkpoint)
        self.model.test(self.X_test, self.y_test)
    
    def predict_ahead(self, X=None, starting_date=None, days_ahead=30):
        dates = []
        lst_output = []
        temp_input = list(X)
        date_increment = 1
        for i in range(days_ahead//self.n_steps_ahead):
            x_input = np.array(temp_input[:])
            x_input = x_input.reshape((1, self.n_steps, 1))
            yhat = self.model.predict(tf.convert_to_tensor(x_input, dtype='float64'))
            yhat = tf.squeeze(yhat) 
            temp_input.extend(yhat)
            temp_input = temp_input[len(yhat):]
            lst_output.extend(yhat)
            for j in range(self.n_steps_ahead):
              dates.append(starting_date + np.timedelta64(date_increment, 'D'))
              date_increment += 1
              print("Predicted date:", dates[-1], lst_output[-1])
        result_df = pd.DataFrame(dict(ds = pd.Series(dates),
                                    yhat = pd.Series(lst_output))) 
        return result_df    
   
    def predict(self, X=None):
        return self.model.predict(X)    
    
    def run(self):
        if not self.load:
            self.train_and_test()
        predictions = self.predict_ahead(self.dataframe.y.values[-self.n_steps:], starting_date=self.dataframe.ds.values[-1], days_ahead=self.days_ahead)
        return predictions