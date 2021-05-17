from tensorflow.keras.layers import Layer, Dropout, LayerNormalization, Conv1D
from .MultiAttention import MultiAttention
from .constants import DROPOUT, FILTER_SIZE, FF_DIM

class TransformerEncoder(Layer):
  def __init__(self, **kwargs):
    super(TransformerEncoder, self).__init__()
    self.attn_heads = list()

  def build(self, input_shape):
    self.attn_multi = MultiAttention()
    self.attn_dropout = Dropout(DROPOUT)
    self.attn_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    self.ff_conv1D_1 = Conv1D(filters=FF_DIM, kernel_size=1, activation='relu')
    self.ff_conv1D_2 = Conv1D(filters=FILTER_SIZE, kernel_size=1) # input_shape[0]=(batch, seq_len, 3), input_shape[0][-1]=3 
    self.ff_dropout = Dropout(DROPOUT)
    self.ff_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)    
  
  def call(self, x): 
    inputs = (x, x, x)
    attn_layer = self.attn_multi(inputs)
    attn_layer = self.attn_dropout(attn_layer)
    attn_layer = self.attn_normalize(inputs[0] + attn_layer)
    ff_layer = self.ff_conv1D_1(attn_layer)
    ff_layer = self.ff_conv1D_2(ff_layer)
    ff_layer = self.ff_dropout(ff_layer)
    ff_layer = self.ff_normalize(inputs[0] + ff_layer)
    return ff_layer 