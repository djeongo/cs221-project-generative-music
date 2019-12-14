# Midi encoder
# Convert Midi into word representation
# p12 wait6, p18 wait6
import logging
import mido
from collections import defaultdict
from mido import MidiFile, MidiTrack, Message
from util import track_to_sequence
import music21
import music21.midi as midi
import music21.converter as converter
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def analyze(txt, outpath, sparsity, normalize=True):
    groups = [set([])]
    with open(txt, 'r') as f:
        for note in f.read().split():
            if note.startswith('w'):
                # groups[-1] = sorted(groups[-1])
                groups.append(set([]))
            else:
                groups[-1].add(note)
    
    n = len(groups)
    matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(i, n):
            max_len = 1
            if normalize:
                max_len = max(len(groups[i]), len(groups[j]))
            if max_len > 0:
                matrix[i][j] = len(groups[i].intersection(groups[j]))/max_len
                matrix[j][i] = matrix[i][j]
    
    plt.clf()
    sns.heatmap(matrix, cmap="Blues")
    plt.tight_layout()
    plt.savefig('{}/{}'.format(outpath, os.path.basename(txt).replace('.txt', '.png')))

    sparsity.append((np.sum(matrix) / np.prod(matrix.shape)))
    

if __name__=="__main__":
    path=sys.argv[1]
    outpath=sys.argv[2]
    sparsity = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.startswith('.') and file.endswith('.txt') and file.endswith('0.txt'):
                print('Processing {}'.format(file))
                try:
                    analyze(os.path.join(root, file), outpath, sparsity)
                except Exception as e:
                    print(e)
                    pass

    np.save(os.path.join(outpath, 'sparsity.csv'), sparsity)
