from .preprocess import prepare_data, split_data, make_arrays_equal_length
from .model import LSTMForecaster
from .constants import N_STEPS, N_FEATURES, MODEL_CHECKPOINT
import tensorflow as tf

# Wrapper around 
class LSTMForecasterBuilder:
    
    def __init__(self, dataframe=None, load=True, load_path=MODEL_CHECKPOINT):
        self.__preprocess_data(dataframe)
        # Load tf model
        if load: self.model = tf.keras.models.load_model(MODEL_CHECKPOINT)
        else: self.model = LSTMForecaster()
    
    # Wrapper of all preprocessing and data splitting
    def __preprocess_data(self, dataframe):
        dataframe.y = dataframe.y.astype('float32')
        self.__extract_train_and_test_data(dataframe.y.values)
    
    # Get train and test sets
    # Convert Xs to correct shape
    def __extract_train_and_test_data(self, values):
        self.X_train, self.y_train, self.X_test, self.y_test = split_data(values, difference=1)
        self.X_train = prepare_data(self.X_train, N_STEPS)
        self.X_test = prepare_data(self.X_test, N_STEPS)
        #self.X_train, self.y_train = make_arrays_equal_length(self.X_train, self.y_train)
        #self.X_test, self.y_test = make_arrays_equal_length(self.X_test, self.y_test)
        self.X_train = self.__reshape_input(self.X_train)
        self.X_test = self.__reshape_input(self.X_test)
        
    def __reshape_input(self, X):
        return X.reshape((X.shape[0], X.shape[1], N_FEATURES))
        
    def train(self, batch_size=64):
        self.model.train(self.X_train, self.y_train, batch_size=batch_size)
    
    def test(self, batch_size=8):
        self.X_test, self.y_test = make_arrays_equal_length(self.X_test, self.y_test)
        self.model.evaluate(self.X_test, self.y_test, batch_size=batch_size)
        
    # Run train and test
    def train_and_test(self):
        self.train()
        self.test()
