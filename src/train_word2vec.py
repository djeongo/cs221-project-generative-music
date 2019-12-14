
from gensim.models import Word2Vec
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from config import *
import argparse
import logging
import numpy as np
import os

max_len = 100

logger = logging.getLogger(__name__)

def get_training_data(example, word_to_index):
    X = []
    y = []
    for i in range(len(example)-max_len):
        X.append([word_to_index[word] for word in example[i:i+max_len]])
        y.append(example[i+max_len])

    y_onehot = onehot_encode(y, word_to_index)

    return X, y_onehot

def read_file(fname):
    with open(fname) as f:
        data = f.read().split()
        return data

def get_embedding_matrix(word2vec_file):
    wv_model = Word2Vec.load(word2vec_file)

    vocab_size = len(wv_model.wv.vocab)

    # one-hot encoding: assign number to words
    words = []
    for word in wv_model.wv.vocab:
        words.append(word)

    embedding_matrix = np.random.rand(vocab_size, EMBEDDING_SIZE)
    
    word_to_index = {}
    for index, word in enumerate(words):
        embedding_matrix[index] = wv_model.wv[word]
        logger.info("word: {}, embedding: {}".format(word, embedding_matrix[index][:10]))
        word_to_index[word] = index

    return embedding_matrix, word_to_index, words

def build_model(embedding_matrix, lstm_layers, lstm_size, drop_out):
    vocab_size = embedding_matrix.shape[0]

    model = Sequential()
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix])

    model.add(embedding_layer)
    for _ in range(lstm_layers-1):
        model.add(LSTM(lstm_size, dropout=drop_out, return_sequences=True))
    model.add(LSTM(lstm_size, dropout=drop_out))
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(
        optimizer='adam',
        metrics=['accuracy'],
        loss='categorical_crossentropy')

    return model

def onehot_encode(y, word_to_index):
    vocab_size = len(word_to_index)
    y_onehot = np.zeros((len(y), vocab_size))
    for i, word in enumerate(y):
        word_index = word_to_index[word]
        y_onehot[i][word_index] = 1

    return y_onehot

def save_words(index_to_word):
    logger.info('Saving index_to_word')
    with open('index_to_word.txt', 'w') as f:
        f.write(' '.join(index_to_word))

def collect_data(word_to_index, path='train-data/'):
    Xs = []
    Ys = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.startswith('.') and file.endswith('.txt'):
                logger.info('Processing {}'.format(file))
                example = read_file(os.path.join(root, file))
                X, Y = get_training_data(example, word_to_index)
                Xs.append(X)
                Ys.append(Y)
                logger.info(len(X))

    X = [x for X in Xs for x in X]
    Y = [y for Y in Ys for y in Y]
    return X, Y

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lstm-layers', default=2, type=int)
    parser.add_argument('--lstm-size', default=128, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--batchsize', default=8192, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    ARGS = parser.parse_args()

    embedding_matrix, word_to_index, index_to_word = get_embedding_matrix(
        'word2vec.model',
        lstm_layers=ARGS.lstm_layers,
        lstm_size=ARGS.lstm_size,
        dropout=ARGS.dropout)
    save_words(index_to_word)

    X, y = collect_data(word_to_index, path='train-data')
    logger.info("Number of training data: {}".format(len(X)))
    
    model = build_model(embedding_matrix)
    X = np.array(X)
    y = np.array(y)

    checkpoint = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1)
    callbacks = [checkpoint]

    model.fit(
        X, y, epochs=ARGS.epoch, callbacks=callbacks, validation_split=0.1,
        batch_size=ARGS.batch_size)