from collections import defaultdict
import mido
import numpy as np
import pandas as pd
import os
import glob, os
import argparse
import logging
import random
from midi_to_words_music21 import decode_notes, notes_to_stream, stream_to_midi
from test_word2vec import sample_note
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def sample(notes):
    print(notes)
    note_names = list(notes.keys())
    counts = [notes[note] for note in note_names]
    probs = counts / np.sum(counts)
    return note_names[sample_note(probs)]

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data')
    parser.add_argument('--n', default=10, type=int)
    parser.add_argument('--len', default=1000, type=int)
    parser.add_argument('--output-path', required=True)
    ARGS = parser.parse_args()

    encoded_notes = []
    for root, dirs, files in os.walk(ARGS.train_data):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file)) as f:
                    encoded_notes.extend(f.read().split())

    transition_matrix = defaultdict(lambda : defaultdict(int))
    
    for n1, n2 in zip(encoded_notes[:-1], encoded_notes[1:]):
        transition_matrix[n1][n2] += 1

    for i in range(ARGS.n):
        print('Generating {}/{}'.format(i, ARGS.n))
        notes = random.sample(transition_matrix.keys(), 1)
        for j in range(ARGS.len):
            prev_note = notes[-1]
            notes.append(sample(transition_matrix[prev_note]))
    
        decoded_notes = decode_notes(notes)
        stream = notes_to_stream(decoded_notes)
        stream_to_midi(stream, os.path.join(ARGS.output_path, '{}.mid'.format(i)) )