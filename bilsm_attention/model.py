import tensorflow as tf
from bilsm_attention.Encoder import  Encoder
from bilsm_attention.Attention import BahdanauAttention
from bilsm_attention.Decoder import Decoder
import numpy as np

class BiLSTM_Attention(tf.keras.Model):
    def __init__(self, embedding_dim, units, nClasses):
        super(BiLSTM_Attention, self).__init__()
        self.encoder = Encoder(embedding_dim, units)
        self.decoder = Decoder(nClasses)
        self.attention = BahdanauAttention(units)

    def call(self, x):
        x = tf.convert_to_tensor(x)
        final_output, state_h = self.encoder(x)
        context_vector, attention_weights = self.attention(final_output, state_h)
        print('atten',context_vector,attention_weights)
        x = self.decoder(context_vector)
        return x
