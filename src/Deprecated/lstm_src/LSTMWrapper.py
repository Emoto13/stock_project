import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model


from .LSTMForecaster import LSTMForecaster
from .PreProcessor import PreProcessor
from .constants import MODEL_CHECKPOINT, N_STEPS, N_FEATURES


class LSTMWrapper(BaseModelWrapper):
    def __init__(self, dataframe=None, periodicity="weekly", 
                 test_mode=False, plot_test=False,
                 load=False, load_path=MODEL_CHECKPOINT,
                 **kwargs):
        super().__init__(dataframe=dataframe, periodicity=periodicity, test_mode=test_mode)
        self.load = load
        self.plot_test = plot_test
        self.days_ahead = kwargs.pop('days_ahead') if 'days_ahead' in kwargs.keys() else 365

        self.__extract_train_and_test_data(self.dataframe.y.values)
        
        # Load tf model
        if load: self.model = load_model(load_path)
        else: self.model = LSTMForecaster(**kwargs)
    
    # Get train and test sets
    # Convert Xs to correct shape
    def __extract_train_and_test_data(self, values):
        self.X, self.y = PreProcessor.prepare_data(values, N_STEPS)
        self.X_train, self.y_train, self.X_test, self.y_test = PreProcessor.split(self.X, self.y)
        
        self.X = self.__reshape_input(self.X)
        self.X_train = self.__reshape_input(self.X_train)
        self.X_test = self.__reshape_input(self.X_test)
        
    def __reshape_input(self, array):
        return array.reshape((array.shape[0], array.shape[1], N_FEATURES))
        
    # Run train and test
    def train_and_test(self):
        self.model.train(self.X_train, self.y_train)
        # Use best model 
        self.model.load_weights()
        if self.test_mode:
            print("test")
            self.model.test(self.X_test, self.y_test)
            if self.plot_test:
              test_res = self.predict(np.reshape(self.X_test, (-1,1))[:N_STEPS], days_ahead=len(self.y_test) - N_STEPS)
              plt.plot(np.reshape(self.y_test, (-1,1)), color = 'red', label = 'Real price')
              plt.plot(test_res.yhat.values, color = 'blue', label = 'Predicted price')
              plt.title('Google price prediction: Test set')
              plt.xlabel('Time')
              plt.ylabel('Price')
              plt.legend()
              plt.show()


    def predict(self, X=None, days_ahead=30, verbose=0):
        last_available_date = self.dataframe.ds.values[-1]
        dates = []
        
        output = []
        temp_predictions = list(X)
        # Insert dummy element
        temp_predictions.insert(0, 0)
        
        for i in range(1, days_ahead):
            lstm_input = np.array(temp_predictions[1:])
            lstm_input = lstm_input.reshape([1, N_STEPS, N_FEATURES])
            yhat = self.model.predict(tf.convert_to_tensor(lstm_input, dtype='float32'))
            temp_predictions.append(yhat[0][0])
            temp_predictions = temp_predictions[1:]
            
            output.append(yhat[0][0].numpy())
            dates.append(last_available_date + np.timedelta64(i, 'D'))
            print("Predicted date:", dates[-1], output[-1])
        result_df = pd.DataFrame(dict(ds = pd.Series(dates),
                                    yhat = pd.Series(output))) 
        return result_df 
    
    def run(self):
        if not self.load:
            self.train_and_test()
        predictions = self.predict(self.dataframe.y.values[:N_STEPS], days_ahead=self.days_ahead)
        return predictions