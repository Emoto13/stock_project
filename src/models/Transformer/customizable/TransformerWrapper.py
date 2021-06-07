import pandas as pd
import numpy as np
from .TransformerForecaster import TransformerForecaster
from .PreProcessor import PreProcessor


class TransformerWrapper:
    def __init__(self, dataframe=None, days_ahead=30, sequence_len=128,
                 d_k=32, d_v=32,n_heads=16, ff_dim=256, filter_size=3, dropout=0.1,
                 epochs=50, batch_size=32, load=False,
                 train_test_split = 0.1, validation_split=0.1,
                 checkpoint='./saved_models/Transformer_time_embeddings_V1.ckpt'):
        self.dataframe = dataframe
        self.sequence_len = sequence_len
        self.load = load
        self.days_ahead = days_ahead
        self.checkpoint = checkpoint
        self.model = TransformerForecaster(sequence_len = sequence_len,
                                           d_k = d_k,
                                           d_v = d_v,
                                           n_heads = n_heads,
                                           ff_dim = ff_dim,
                                           filter_size = filter_size,
                                           dropout = dropout,
                                           epochs = epochs,
                                           batch_size = batch_size,
                                           validation_split=validation_split,
                                           checkpoint=self.checkpoint)
      
        self.X, self.y = PreProcessor.prepare_data(self.dataframe.y.values, sequence_len)
        self.X_train, self.y_train, self.X_test, self.y_test = PreProcessor.split(self.X, self.y, train_test_split)
        
 

    def train_and_test(self):
        self.model.train(self.X_train, self.y_train)
        # Use best model 
        self.model.load_weights()
        self.model.test(self.X_test, self.y_test)
        
    def predict_ahead(self, X=None, starting_date=None ,days_ahead=30):
        dates = []
        output = []
        temp_predictions = list(X)
        # Insert dummy element
        temp_predictions.insert(0, 0)
        print(X)
        
        for i in range(1, days_ahead):
            lstm_input = np.array(temp_predictions[1:], dtype='float64')
            lstm_input = lstm_input.reshape([1, self.sequence_len, 1])
            yhat = self.model.predict(tf.convert_to_tensor(lstm_input, dtype='float64'))
            temp_predictions.append(yhat[0][0])
            temp_predictions = temp_predictions[1:]
            print(yhat)
            output.append(yhat[0].numpy()[0])
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
        predictions = self.predict_ahead(self.dataframe.y.values[-self.sequence_len:], starting_date=self.dataframe.ds.values[-1], days_ahead=self.days_ahead)
        return predictions
