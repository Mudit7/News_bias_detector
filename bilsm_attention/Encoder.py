import tensorflow as tf

class Encoder(tf.keras.Model):
  def __init__(self, embedding_dim, enc_units):
    super(Encoder, self).__init__()
    self.enc_units = enc_units
    self.mask = tf.keras.layers.Masking(mask_value=0)
    self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM
                                     (self.enc_units,
                                      dropout=0.3,
                                      return_sequences=True,
                                      return_state=True,
                                      recurrent_activation='tanh',
                                      recurrent_initializer='glorot_uniform'),merge_mode='concat', name="bi_lstm_0")


  def call(self, x):
      masked_inputs = self.mask(x)
      final_output, forward_h, forward_c, backward_h, backward_c = self.bilstm(masked_inputs)
      state_h = tf.concat([forward_h, backward_h],axis = -1)
      return final_output, state_h
