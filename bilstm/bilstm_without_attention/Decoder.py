import tensorflow as tf

class Decoder(tf.keras.Model):
  def __init__(self,nclasses):
    super(Decoder, self).__init__()
    self.fc = tf.keras.layers.Dense(nclasses, activation='sigmoid')

  def call(self, context_vector):
    x = self.fc(context_vector)
    return x