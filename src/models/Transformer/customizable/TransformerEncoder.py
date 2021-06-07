from tensorflow.keras.layers import Layer, Dropout, LayerNormalization, Conv1D
from .MultiAttention import MultiAttention

class TransformerEncoder(Layer):
  def __init__(self, d_k=32, d_v=32, n_heads=16, ff_dim=256, filter_size=3, dropout=0.1):
    super(TransformerEncoder, self).__init__()
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.ff_dim = ff_dim
    self.filter_size = filter_size
    self.dropout = dropout
    self.attn_heads = []


  def build(self, input_shape):
    self.attn_multi = MultiAttention(d_k=self.d_k,
                                     d_v=self.d_v,
                                     n_heads=self.n_heads,
                                     filter_size=self.filter_size)
    self.attn_dropout = Dropout(self.dropout)
    self.attn_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    self.ff_conv1D_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
    self.ff_conv1D_2 = Conv1D(filters=self.filter_size, kernel_size=1) # input_shape[0]=(batch, seq_len, 3), input_shape[0][-1]=3 
    self.ff_dropout = Dropout(self.dropout)
    self.ff_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)    
  
  def call(self, x): # inputs = (in_seq, in_seq, in_seq)
    inputs = (x, x, x)
    attn_layer = self.attn_multi(inputs)
    attn_layer = self.attn_dropout(attn_layer)
    attn_layer = self.attn_normalize(inputs[0] + attn_layer)
    ff_layer = self.ff_conv1D_1(attn_layer)
    ff_layer = self.ff_conv1D_2(ff_layer)
    ff_layer = self.ff_dropout(ff_layer)
    ff_layer = self.ff_normalize(inputs[0] + ff_layer)
    return ff_layer 