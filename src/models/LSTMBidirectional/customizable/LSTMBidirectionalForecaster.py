from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Bidirectional
import tensorflow as tf


class LSTMForecaster:
    def __init__(self, units=128, n_steps=7, neurons=3, activation='relu', epochs=35, batch_size=32, dropout=0.0, clipnorm=0, checkpoint="LSTM_checkpoint"):
        # Deactivating CUDA because tensorflow throws Unknown error for LSTM Networks
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.units = units
        self.n_steps = n_steps
        self.neurons = neurons
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.clipnorm = clipnorm
        self.checkpoint = checkpoint
        self.model = self.__build_model()                      
    
    def __build_model(self):
        model = Sequential([
              Bidirectional(LSTM(units=self.units, activation=self.activation, return_sequences=True)),
              Dropout(self.dropout)
        ] * self.neurons)
        model.add(Dense(1)) 
        optimizer = Adam(clipnorm=self.clipnorm)
        model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mse', 'mape'])
        return model

    def train(self, X_train=None, y_train=None):
        #stop_early = EarlyStopping(monitor='val_mse', patience=int(0.5*self.epochs))
        cp_callback = ModelCheckpoint(filepath=self.checkpoint,
                                      save_weights_only=True,
                                      save_best_only=True,
                                      verbose=1)

        self.model.fit(X_train, y_train,
                       epochs=self.epochs, verbose=1,
                       batch_size=self.batch_size,
                       callbacks=[cp_callback],#, stop_early],
                       validation_split = 0.1)

    def test(self, X_test=None, y_test=None):
        return self.model.evaluate(x=X_test, y=y_test, verbose=1)
    
    def predict(self, X=None):
        return self.model(X, training=False)
        
    def load_weights(self, checkpoint):
        return self.model.load_weights(checkpoint)