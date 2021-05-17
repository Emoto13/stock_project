from tensorflow.keras.layers import Layer, Dense
import tensorflow as tf

from .SingleAttention import SingleAttention
from .constants import FILTER_SIZE, N_HEADS

class MultiAttention(Layer):
  def __init__(self):
    super(MultiAttention, self).__init__()

  def build(self, input_shape):
    self.attn_heads = [SingleAttention()] * N_HEADS
    self.linear = Dense(FILTER_SIZE, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')

  def call(self, inputs):
    attn = [self.attn_heads[i](inputs) for i in range(N_HEADS)]
    concat_attn = tf.concat(attn, axis=-1)
    multi_linear = self.linear(concat_attn)
    return multi_linear 