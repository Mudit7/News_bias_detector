import tensorflow as tf
from bilstm.bilsm_attention.model import BiLSTM_Attention
from keras_preprocessing import sequence
import numpy as np
from gensim.models import Word2Vec
import pandas as pd
import gensim.downloader as api
from tensorflow.keras.preprocessing.text import text_to_word_sequence

# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(vocab_size, 64),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#     tf.keras.layers.Attention(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])

def dataloader(w2v,max_len,tosave=False):
    def preprocess(text):
        embeddings = []
        words = text_to_word_sequence(text)
        for word in words:
            if word in w2v:
                embeddings.append(w2v[word])

        cur_seq_len = len(embeddings)
        if cur_seq_len < max_len:
            embeddings = np.pad(embeddings, [(0, max_len - cur_seq_len), (0, 0)])
        else:
            embeddings = embeddings[cur_seq_len - max_len :]

        return embeddings

    df_file = 'cleaned_data.csv'
    df = pd.read_csv(df_file)
    x_train = []
    y_train = []
    for ind in df.index:
        x_train.append(preprocess(df['texts'][ind]))
        labels = (0,1) if df['labels'][ind] else (1,0)
        y_train.append(labels)

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    if tosave:
        np.savez('processed_data.npz', x_train, y_train)
        print('saved.')
    print('loaded.')
    return x_train,y_train

if __name__ == '__main__':
    # preprocessing
    # corpus = api.load('text8')
    # word2vec = Word2Vec(corpus)
    # word2vec.save('gensim_w2v.emb')
    max_seq_len = 150
    pad_id = 0
    emb_dim = 100
    latent_dim = 90
    no_classes = 2
    load_saved = True

    if not load_saved:
        word2vec = Word2Vec.load('gensim_w2v.emb')
        X,Y = dataloader(word2vec.wv,max_len=max_seq_len,tosave=True)
    else:
        data = np.load('processed_data.npz',allow_pickle=True)
        X = data['arr_0']
        Y = data['arr_1']
    print(f"emb shape = {X.shape}, \nlabel_shape = {Y.shape}")
    # total_size = X.shape[0]
    # train,test = (0.7,0.3)*total_size
    #
    # x_train = X[0:train]
    # y_train = Y[0:train]
    #
    # x_test = X[train:total_size]
    # y_test = X[train:total_size]

    model = BiLSTM_Attention(embedding_dim=emb_dim, units=latent_dim, nClasses=no_classes)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.save_weights('lstm_with_attention')
    model.fit(X,Y,epochs=20,verbose=2,batch_size=30,validation_split=0.3)
    # print(model.predict(x_train[0:20]),y_train[0:20])
