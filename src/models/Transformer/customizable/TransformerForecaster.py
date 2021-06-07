from tensorflow.keras.layers import Concatenate, GlobalAveragePooling1D, Dropout, Dense, Input
from tensorflow.keras.optimizers import  Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from .TransformerEncoder import TransformerEncoder
from .Time2Vector import Time2Vector

class TransformerForecaster:
    def __init__(self, sequence_len=128, d_k=32, d_v=32,
                 n_heads=16, ff_dim=256, filter_size=3, dropout=0.1,
                 epochs=50, batch_size=32, validation_split=0.1,
                 checkpoint='./saved_models/Transformer_time_embeddings_V{VERSION}.ckpt'):
        self.sequence_len = sequence_len
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.filter_size = filter_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.checkpoint = checkpoint
        self.model = self.__create_model()

    def __create_model(self):
        '''Construct model'''
        in_seq = Input(shape=(self.sequence_len, 1))
        x = Time2Vector()(in_seq)
        x = Concatenate(axis=-1)([in_seq, x])

        transformer_encoder = TransformerEncoder(d_k=self.d_k,
                                                 d_v=self.d_v,
                                                 n_heads=self.n_heads, 
                                                 ff_dim=self.ff_dim,
                                                 filter_size=self.filter_size,
                                                 dropout=self.dropout)
        x = transformer_encoder(x)
        x = transformer_encoder(x)
        x = transformer_encoder(x)
        x = GlobalAveragePooling1D(data_format='channels_first')(x)
        x = Dropout(self.dropout)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        out = Dense(1, activation='linear')(x)

        model = Model(inputs=in_seq, outputs=out)
        return model
    
    def train(self, X_train=None, y_train=None):
        cp_callback = ModelCheckpoint(filepath=self.checkpoint,
                                save_best_only=True,
                                save_weights_only=True,
                                verbose=1)
        optimizer = Adam()

        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
        self.model.fit(X_train, y_train,
          epochs=self.epochs, verbose=1,
          batch_size=self.batch_size,
          callbacks=[cp_callback],
          validation_split=self.validation_split)
    
    def test(self, X_test=None, y_test=None):
        return self.model.evaluate(x=X_test, y=y_test, verbose=1)
    
    def predict(self, X=None):
        return self.model(X, training=False)
    
    def load_weights(self):
        return self.model.load_weights(self.checkpoint)