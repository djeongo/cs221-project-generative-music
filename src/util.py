import logging
import mido
import music21
import numpy as np
import os
from collections import defaultdict
from mido import MidiFile, MidiTrack, Message

logger = logging.getLogger(__name__)

NOTE_NAMES = ['C','C#/Db','D','D#/Eb','E','F','F#/Gb','G','G#/Ab','A','A#/Bb','B']
SHORT_NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
# def number_to_note(note_number, short=True):
#     note_names = NOTE_NAMES
#     if short:
#         note_names = SHORT_NOTE_NAMES
#     octave = note_number // 12 - 2
#     name = note_names[note_number % 12]
#     note_name = '{}{}'.format(name, octave+1)
#     return octave, name, note_name


NAMES = {
    'C':0, 
    'C#':1, #
    'D-':1,
    'D':2,
    'D#':3, #
    'E-':3,
    'E':4,
    'F':5,
    'F#':6, #
    'G-':6,
    'G':7,
    'G#':8,
    'A-':8,
    'A':9,
    'A#':10, #
    'B-':10,
    'B':11,
}

PITCHES = [
    'C',
    'C#/D-',
    'D',
    'D#/E-',
    'E',
    'F',
    'F#/G-',
    'G',
    'G#/A-',
    'A',
    'A#/B-',
    'B'
]

def note_to_number(note):
    LOWEST = 12
    name = note[:-1]
    octave = int(note[-1])
    return NAMES[name] + (octave * 12) + LOWEST

def number_to_note(number):
    return number

def get_highest_note(notes):
    note_numbers = [note_to_number(note) for note in notes]
    index = np.argmax(note_numbers)
    return notes[index]

def get_notes_by_offsets(notes):
    offsets = sorted(set([note['offset'] for note in notes]))
    # Collect notes in the same offset
    notes_by_offset = defaultdict(list)
    for note in notes:
        notes_by_offset[note['offset']].append(note['nameWithOctave'])
    return offsets, notes_by_offset

def get_melody(stream):
    notes = stream_to_notes(stream)
    offsets, notes_by_offset = get_notes_by_offsets(notes)
    melody = []
    for offset in offsets:
        melody.append(get_highest_note(notes_by_offset[offset]))
    return melody

def stream_to_notes(stream):
    notes = []

    # new_stream = music21.stream.Stream()
    # print("notesAndRests: {}, notes:{}".format(len(stream.notesAndRests), len(stream.notes)))
    
    for filter in ['Chord', 'Note']:
        noteFilter = music21.stream.filters.ClassFilter(filter)
        for note in stream.recurse().addFilter(noteFilter):
            if note.isNote:
                # print("offset:{} {}: nameWithOctave{} duration:{} duration.quarterLength:{} duration.type:{}"
                # .format(note.offset, note, note.nameWithOctave, note.duration, note.duration.quarterLength,
                # note.duration.type))
                # new_stream.insert(note.offset, music21.note.Note(nameWithOctave=note.nameWithOctave,
                                                    # quarterLength=note.duration.quarterLength))
                notes.append({
                    'offset':note.offset,
                    'nameWithOctave':note.nameWithOctave})
            elif note.isChord:
                # print("offset:{} chord: {}, duration.quarterLength{}".format(note.offset, note, note.duration.quarterLength))
                for p in note.pitches:
                    notes.append({
                        'offset':note.offset,
                        'nameWithOctave':p.nameWithOctave
                    })
                # new_stream.insert(note.offset, music21.chord.Chord([p.nameWithOctave for p in note.pitches]))
            else:
                print("offset:{} rest: duration:{}".format(note.offset, note.duration))
                # new_stream.insert(note.offset, music21.note.Rest(quarterLength = note.duration.quarterLength))
        
    # logger.info("Writing to midi file")
    # mf = midi.translate.streamToMidiFile(new_stream)
    # mf.open('output-stream.mid', 'wb')
    # mf.write()
    # mf.close()

    # sort
    notes = sorted(notes, key=lambda x: x['offset'])

    return notes

def load_words(path='index_to_word.txt'):
    print('Loading index_to_word')
    with open(path) as f:
        return f.read().split()

def melody_to_midi(melody, midi_name='song.mid'):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    time = 32 
    for note in melody:
        track.append(Message('note_on', note=note, velocity=64, time=time))
        track.append(Message('note_off', note=note, velocity=64, time=time))

    mid.save(midi_name)
    
def track_to_sequence(track):
    note_sequence = []
    
    for msg in track:
        if msg.type == 'note_on':
            note = number_to_note(msg.note)
            logger.debug("msg:{}, note:{}".format(msg, note))
        else:
            logger.debug("msg:{}".format(msg))
        note_sequence.append(msg)
    return note_sequence

def iterate_files(path, endswith=".midi", melody_track=1):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(endswith):
                if endswith == ".midi":
                    song_name, _ = os.path.splitext(os.path.basename(file))
                    logger.info(song_name)
                    midi_file = os.path.join(root,file)
                    mid = mido.MidiFile(midi_file)
                    logger.info('ticks_per_beat: {}'.format(mid.ticks_per_beat))

                    for i, track in enumerate(mid.tracks):
                        logger.debug('Track {}: name:{} len:{}'.format(i, track.name, len(track)))
                        sequence = track_to_sequence(track)
                    #sequence = track_to_sequence(mid.tracks[melody_track])
                    yield root, song_name, sequence, mid.ticks_per_beat
                elif endswith == ".roll":
                    yield root, file
