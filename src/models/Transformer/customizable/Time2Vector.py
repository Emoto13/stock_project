from tensorflow.keras.layers import Layer
import tensorflow as tf

from .constants import SEQ_LEN

class Time2Vector(Layer):
  def __init__(self, **kwargs):
    super(Time2Vector, self).__init__()

  def build(self, input_shape):
    self.weights_linear = self.add_weight(name='weight_linear',
                                shape=(int(SEQ_LEN),),
                                initializer='uniform',
                                trainable=True)
    
    self.bias_linear = self.add_weight(name='bias_linear',
                                shape=(int(SEQ_LEN),),
                                initializer='uniform',
                                trainable=True)
    
    self.weights_periodic = self.add_weight(name='weight_periodic',
                                shape=(int(SEQ_LEN),),
                                initializer='uniform',
                                trainable=True)

    self.bias_periodic = self.add_weight(name='bias_periodic',
                                shape=(int(SEQ_LEN),),
                                initializer='uniform',
                                trainable=True)

  def call(self, x):
    x = tf.squeeze(x, axis=(2,)) # Convert (batch, seq_len, 5) to (batch, seq_len)
    time_linear = self.weights_linear * x + self.bias_linear
    time_linear = tf.expand_dims(time_linear, axis=-1) # (batch, seq_len, 1)
    time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
    time_periodic = tf.expand_dims(time_periodic, axis=-1)
    return tf.concat([time_linear, time_periodic], axis=-1) # (batch, seq_len, 2)