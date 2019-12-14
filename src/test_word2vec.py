import argparse
from keras.models import load_model
from midi_to_words_music21 import decode_notes, stream_to_midi, notes_to_stream
import numpy as np
import os
import random
from util import load_words
from config import * 

def sample_note(y_predict, top_n=None):
    # print(y_predict)
    p = np.random.rand()

    # select top_n
    indices = np.flip(np.argsort(y_predict))[:top_n]
    y_sorted = np.flip(np.sort(y_predict))[:top_n]

    # normalize
    y_sorted = y_sorted / np.sum(y_sorted)
    
    # sort y values by prob        
    y_cumsum = np.cumsum(y_sorted)
    i_selected = None
    for i in range(len(indices)):
        if p < y_cumsum[i]:
            i_selected = i
            break

    print('sampled: {} with prob: {}'.format(
        indices[i_selected], y_sorted[i_selected]))
    return indices[i_selected]

def notes_to_midi(x, index_to_word, fn):
    notes = [index_to_word[index] for index in x]
    
    notes_decoded = decode_notes(notes)
    
    stream_from_decoding = notes_to_stream(notes_decoded)
    stream_to_midi(stream_from_decoding, fn)

def generate(ARGS):
    model = load_model(ARGS.model_file)
    index_to_word = load_words(ARGS.index_to_word)

    word_to_index = {}
    for index, word in enumerate(index_to_word):
        word_to_index[word] = index

    x = []
    if ARGS.seed:
        with open(ARGS.seed) as f:
            notes = f.read().split()
        x = [word_to_index[note] for note in notes[:ARGS.seed_len]]
        notes_to_midi(
            x,
            index_to_word,
            os.path.join(ARGS.output_path,'seed-{}.midi'.format(ARGS.tag)))
    else:
        x = [random.randint(0, len(index_to_word)-1)]

    for i in range(ARGS.n):
        print('{}/{}: last 10 notes:{}'.format(i, ARGS.n, np.array(x[-10:])))
        y_predict = model.predict(np.array([x])).flatten()
        if ARGS.sample:
            x.append(sample_note(y_predict, ARGS.top_n))
        else:
            x.append(np.argmax(y_predict))

    out_file = 'generated-{}.mid'.format(ARGS.tag)
    print('Writing to {}'.format(out_file))
    notes_to_midi(
        x,
        index_to_word,
        os.path.join(ARGS.output_path, out_file))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', default='weights.45.hdf5')
    parser.add_argument('--sample', default=False, action='store_true')
    parser.add_argument('--top-n', default=None, type=int)
    parser.add_argument('--n', default=None, type=int)
    parser.add_argument('--seed', default=None)
    parser.add_argument('--seed-len', default=100, type=int)
    parser.add_argument('--tag', default='')
    parser.add_argument('--output-path', default='./')
    parser.add_argument('--index-to-word', default='./index_to_word.txt')
    ARGS = parser.parse_args()

    generate(ARGS)