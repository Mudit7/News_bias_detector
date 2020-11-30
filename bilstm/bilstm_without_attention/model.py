import tensorflow as tf

from bilstm.bilstm_without_attention.Encoder import Encoder
from bilstm.bilstm_without_attention.Decoder import Decoder
import numpy as np


class BiLSTM_without_Attention(tf.keras.Model):
    def __init__(self, embedding_dim, units, nClasses):
        super(BiLSTM_without_Attention, self).__init__()
        self.encoder = Encoder(embedding_dim, units)
        self.decoder = Decoder(nClasses)

    def call(self, x):
        x = tf.convert_to_tensor(x)
        final_output, state_h = self.encoder(x)
        x = self.decoder(state_h)
        return x
