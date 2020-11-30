import tensorflow as tf
import numpy as np
max_len = 150
rnn_cell_size = 128
vocab_size=250

class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

emb_dim = 100
sequence_input = tf.keras.layers.Input(shape=(max_len,emb_dim), dtype='float')

lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM
                                     (rnn_cell_size,
                                      dropout=0.3,
                                      return_sequences=True,
                                      return_state=True,
                                      recurrent_activation='relu',
                                      recurrent_initializer='glorot_uniform'), name="bi_lstm_0")(sequence_input)

lstm, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional \
    (tf.keras.layers.LSTM
     (rnn_cell_size,
      dropout=0.2,
      return_sequences=True,
      return_state=True,
      recurrent_activation='relu',
      recurrent_initializer='glorot_uniform'))(lstm)

state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])

#  PROBLEM IN THIS LINE
context_vector, attention_weights = Attention(rnn_cell_size)(lstm, state_h)

output = tf.keras.layers.Dense(2, activation='sigmoid')(context_vector)

model = tf.keras.Model(inputs=sequence_input, outputs=output)

# summarize layers
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#prepare input
data = np.load('processed_data.npz',allow_pickle=True)
x_train = data['arr_0']
y_train = data['arr_1']

history = model.fit(x_train,
                    y_train,
                    epochs=5,
                    batch_size=20,verbose=2)