from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from ...BaseForecaster import BaseForecaster
from .constants import EPOCHS, BATCH_SIZE, MODEL_CHECKPOINT

class LSTMForecaster(BaseForecaster):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Deactivating CUDA because tensorflow throws Unknown error for LSTM Networks
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.model = self.__build_model()                      
    
    def __build_model(self):
        model = Sequential([
              LSTM(units=32, activation='relu', return_sequences=True),
              LSTM(units=64, activation='relu', return_sequences=True),
              LSTM(units=128, activation='relu', return_sequences=True),
              LSTM(units=256, activation='relu', return_sequences=True),
              LSTM(units=512, activation='relu', return_sequences=True),
        ])
       
        model.add(Dense(1)) 
        return model

    def train(self, X_train=None, y_train=None):
        cp_callback = ModelCheckpoint(filepath=MODEL_CHECKPOINT,
                                      save_weights_only=True,
                                      save_best_only=True,
                                      verbose=1)
        lr = Adam()
        self.model.compile(optimizer=lr, loss='mae', metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mse', 'mape'])

        self.model.fit(X_train, y_train,
                       epochs=EPOCHS, verbose=1,
                       batch_size=BATCH_SIZE,
                       callbacks=[cp_callback],
                       validation_split = 0.05
                       )

        self.model.save_weights(MODEL_CHECKPOINT)

    def test(self, X_test=None, y_test=None):
        return self.model.evaluate(x=X_test, y=y_test, verbose=1)
    
    def predict(self, X=None):
        return self.model(X, training=False)
        
    def load_weights(self):
        return self.model.load_weights(MODEL_CHECKPOINT)