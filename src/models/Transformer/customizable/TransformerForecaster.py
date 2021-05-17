from tensorflow.keras.layers import Concatenate, GlobalAveragePooling1D, Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from .constants import SEQ_LEN, DROPOUT, BATCH_SIZE, EPOCHS, MODEL_CHECKPOINT
from .TransformerEncoder import TransformerEncoder
from .Time2Vector import Time2Vector


class TransformerForecaster:
    def __init__(self, **kwargs):
        super(TransformerForecaster, self).__init__(**kwargs)
        self.model = self.__create_model()

    def __create_model(self):
        '''Construct model'''
        in_seq = Input(shape=(SEQ_LEN, 1))
        x = Time2Vector()(in_seq)
        x = Concatenate(axis=-1)([in_seq, x])
        x = TransformerEncoder()(x)
        x = TransformerEncoder()(x)
        x = TransformerEncoder()(x)
        x = GlobalAveragePooling1D(data_format='channels_first')(x)
        x = Dropout(DROPOUT)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(DROPOUT)(x)
        out = Dense(1, activation='linear')(x)

        model = Model(inputs=in_seq, outputs=out)
        return model
    
    def train(self, X_train=None, y_train=None):
        cp_callback = ModelCheckpoint(filepath=MODEL_CHECKPOINT,
                                save_weights_only=True,
                                save_best_only=True,
                                verbose=1
                              )
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
        self.model.fit(X_train, y_train,
          epochs=EPOCHS, verbose=1,
          batch_size=BATCH_SIZE,
          callbacks=[cp_callback],
          validation_split = 0.1)
    
    def test(self, X_test=None, y_test=None):
        return self.model.evaluate(x=X_test, y=y_test, verbose=1)
    
    def predict(self, X=None):
        return self.model(X, training=False)
    
    def load_weights(self):
        return self.model.load_weights(MODEL_CHECKPOINT)