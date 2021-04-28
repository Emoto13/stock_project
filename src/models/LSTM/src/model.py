from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

from .constants import N_STEPS, N_FEATURES, MODEL_CHECKPOINT

class LSTMForecaster:
    def __init__(self):
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(N_STEPS, N_FEATURES)))
        self.model.add(LSTM(50, activation='relu'))
        self.model.add(Dense(1))
    
    def train(self, X_train=None, y_train=None, batch_size=32):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_CHECKPOINT,
                                                         save_best_only=True,
                                                         verbose=1)
        self.model.compile(optimizer='adam', loss='mae')
        self.model.fit(X_train, y_train,
                       epochs=300, verbose=1,
                       batch_size=batch_size,
                       callbacks=[cp_callback],
                       validation_split = 0.1)

    def evaluate(self, X_test=None, y_test=None, batch_size=8):
        return self.model.evaluate(x=X_test, y=y_test, batch_size=batch_size, verbose=1)
    
    def predict(self, X=None):
        return self.model.predict(X)
    