# Midi encoder
# Convert Midi into word representation
# p12 wait6, p18 wait6
import logging
import mido
from collections import defaultdict
from mido import MidiFile, MidiTrack, Message
from util import track_to_sequence, stream_to_notes, get_notes_by_offsets
import music21
import music21.midi as midi
import music21.converter as converter
import numpy as np
import os
import argparse
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def notes_to_stream(notes):
    new_stream = music21.stream.Stream()

    for note in notes:
        print(note)
        new_stream.insert(
            note['offset'],
            music21.note.Note(nameWithOctave=note['nameWithOctave']))

    return new_stream
    
def stream_to_midi(stream, midi_file):
    print('Writing to {}'.format(midi_file))
    mf = midi.translate.streamToMidiFile(stream)
    mf.open(midi_file, 'wb')
    mf.write()
    mf.close()

def encode_notes(notes):
    offsets, notes_by_offset = get_notes_by_offsets(notes)

    encoding = []

    for offset1, offset2 in zip(offsets[:-1], offsets[1:]):
        # print all notes
        for note in notes_by_offset[offset1]:
            encoding.append(note)
        # print wait
        encoding.append('w{}'.format(np.round(float(offset2-offset1), 3)))

    return encoding

from fractions import Fraction

def decode_notes(encoding):
    offset = 0
    notes = []
    for encoded in encoding:
        if encoded.startswith('w'):
            print(encoded)
            offset += Fraction(encoded[1:])
        else:
            notes.append({'offset':offset, 'nameWithOctave':encoded})
    return notes

def encoding_to_file(encoding, fname):
    with open(fname, 'w') as f:
        f.write(' '.join(encoding))
    print('Finished writing encoding to {}'.format(fname))

def encode_midi(midi_file, transpose=0, outpath='train-data-bach'):
    stream = music21.converter.parse(midi_file).transpose(transpose)
    notes = stream_to_notes(stream)

    stream_from_notes = notes_to_stream(notes)
    stream_to_midi(stream_from_notes, 'from_notes.midi')

    encoding = encode_notes(notes)

    notes_decoded = decode_notes(encoding)

    stream_from_decoding = notes_to_stream(notes_decoded)
    stream_to_midi(stream_from_decoding, 'from_decoding.midi')

    print(len(encoding))
    print(len(set(encoding)))

    encoding_fname = os.path.basename(midi_file).replace(
        '.mid', '{}.txt'.format(transpose))
    encoding_to_file(encoding, os.path.join(outpath, encoding_fname))

if __name__=="__main__":
    path=sys.argv[1] #'batch-generated-bach/'
    outpath=sys.argv[2] #'batch-generated-bach-txt/'
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.startswith('.') and file.endswith('.mid'):
                for transpose in range(-5,7):
                    print('Processing {}, transpose:{}'.format(file, transpose))
                    try:
                        encode_midi(os.path.join(root, file), transpose, outpath)
                    except Exception as e:
                        print(e)