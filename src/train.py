
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from config import *
import argparse
import json
import logging
import numpy as np
import os
import pickle

logger = logging.getLogger(__name__)

def get_training_data(example, word_to_index, max_len):
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

def get_embedding_matrix(word2vec_file='wv.pickle'):
    wv = pickle.load(open(word2vec_file, 'rb'))

    vocab_size = len(wv)

    # one-hot encoding: assign number to words
    words = []
    for word in wv:
        words.append(word)

    embedding_matrix = np.random.rand(vocab_size, EMBEDDING_SIZE)
    
    word_to_index = {}
    for index, word in enumerate(words):
        embedding_matrix[index] = wv[word]
        logger.info("word: {}, embedding: {}".format(word, embedding_matrix[index][:10]))
        word_to_index[word] = index

    return embedding_matrix, word_to_index, words

def build_model(embedding_matrix, lstm_layers, lstm_size, dropout, activation='softmax'):
    vocab_size = embedding_matrix.shape[0]

    model = Sequential()
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix])

    model.add(embedding_layer)
    for _ in range(lstm_layers-1):
        model.add(LSTM(lstm_size, dropout=dropout, return_sequences=True))
    model.add(LSTM(lstm_size, dropout=dropout))
    model.add(Dense(vocab_size, activation=activation))

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

def collect_data(word_to_index, path='train-data/', max_len=100):
    Xs = []
    Ys = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.startswith('.') and file.endswith('.txt'):
                logger.info('Processing {}'.format(file))
                example = read_file(os.path.join(root, file))
                X, Y = get_training_data(example, word_to_index, max_len)
                Xs.append(X)
                Ys.append(Y)
                logger.info(len(X))

    X = [x for X in Xs for x in X]
    Y = [y for Y in Ys for y in Y]
    return X, Y

class CodaLabStats(Callback):
    def on_train_begin(self, log={}):
        pass

    def on_epoch_end(self, batch, logs={}):
        loss = logs.get('loss')
        acc = logs.get('acc')
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_acc')
        stats = {
            'loss':'{}'.format(loss),
            'acc':'{}'.format(acc),
            'val_loss':'{}'.format(val_loss),
            'val_acc':'{}'.format(val_acc)
        }
        json.dump(stats, open('stats.json', 'w'))

def train(ARGS):
    embedding_matrix, word_to_index, index_to_word = get_embedding_matrix(
        ARGS.wv_file)
    save_words(index_to_word)

    X, y = collect_data(
        word_to_index, path=ARGS.train_data, max_len=ARGS.max_len)
    logger.info("Number of training data: {}".format(len(X)))
    
    model = build_model(embedding_matrix,
        lstm_layers=ARGS.lstm_layers,
        lstm_size=ARGS.lstm_size,
        dropout=ARGS.dropout)
    X = np.array(X)
    y = np.array(y)

    checkpoint = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1)
    codalab_stats = CodaLabStats()
    callbacks = [checkpoint, codalab_stats]

    model.fit(
        X, y, epochs=ARGS.epochs, callbacks=callbacks, validation_split=0.1,
        batch_size=ARGS.batch_size)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lstm-layers', default=2, type=int)
    parser.add_argument('--lstm-size', default=128, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--batch-size', default=4096, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--max-len', default=100, type=int)
    parser.add_argument('--wv-file', default='src/wv.pickle')
    parser.add_argument('--train-data', default='train-data')
    ARGS = parser.parse_args()

    logger.info("ARGS: {}".format(ARGS))

    train(ARGS)
    