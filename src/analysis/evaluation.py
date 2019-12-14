from abc import ABC, abstractmethod
from collections import defaultdict
from util import stream_to_notes, get_notes_by_offsets
import matplotlib.pyplot as plt
import argparse
import mido
import music21
import numpy as np
import os
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
import util
from mido import MidiFile, MidiTrack, Message
import logging

MIN_PITCH_COUNT = 0
MAX_PITCH_COUNT = 120

logger = logging.getLogger(__name__)

class Evaluation(ABC):
    def __init__(self):
        pass
        
    @abstractmethod
    def evaluate(self, composition):
        pass

    def log(self):
        pass

class PitchCount(Evaluation):
    
    def __init__(self):
        self.name = 'PitchCount'
        self.feature = None
        self.feature_type = 'scalar'

    def evaluate(self, stream):
        # Count the number of different pitches within a sample
        # Output: scalar
        self.feature = len(set(stream.pitches))

    def log(self):
        print("{}:{}".format(self.name, self.feature))

class PitchClassHistogram(Evaluation):
    def __init__(self):
        self.name = 'PitchClassHistogram'
        self.feature = defaultdict(int)
        self.feature_type = 'histogram'
    
    def evaluate(self, stream):
        # Octave-independent representation of the pitch content with
        # a dimensionality of 12

        for pitch in stream.pitches:
            self.feature[pitch.name] += 1
    
    def log(self):
        print(self.feature)

class PitchClassTransitionMatrix(Evaluation):
    def __init__(self):
        self.name = "PitchClassTransitionMatrix"
        self.feature = defaultdict(int)
        self.feature_type = 'histogram'

    def evaluate(self, stream):
        melody = util.get_melody(stream)

        for p1, p2 in zip(melody[:-1], melody[1:]):
            self.feature[(p1[:-1], p2[:-1])] += 1
            
    def log(self):
        print(self.feature)
        

class PitchRange(Evaluation):
    def __init__(self):
        self.name = "PitchRange"
        self.feature = None
        self.feature_type = 'scalar'

    def evaluate(self, stream):
        # Output: scalar = highest pitch - lowest pitch
        melody = util.get_melody(stream)
        numbers = [util.note_to_number(note) for note in melody]
        self.feature = (max(numbers) - min(numbers))

    def log(self):
        print("range: {}".format(self.feature))

class AveragePitchInterval(Evaluation):
    def __init__(self):
        self.name = "AveragePitchInterval"
        self.feature = None
        self.feature_type = 'scalar'

    def evaluate(self, stream):
        # Average value of the interval between two consecutive pitches in
        # semi-tones.
        # Output: scalar
        intervals = []
        melody = util.get_melody(stream)
        for p1, p2 in zip(melody[:-1], melody[1:]):
            intervals.append(np.abs(util.note_to_number(p1) - util.note_to_number(p2)))

        self.feature = np.mean(intervals)

    def log(self):
        print('average-interval: {}'.format(self.feature))
        

def evaluate(stream):
    _evaluators = [PitchCount(), \
            PitchClassHistogram(), \
            PitchClassTransitionMatrix(), \
            PitchRange(), \
            AveragePitchInterval()]

    evaluators = {}

    for evaluator in _evaluators:
        evaluator.evaluate(stream)
        evaluators[evaluator.name] = evaluator
        # evaluator.log()

    return evaluators
 

def process_scalar_features(evaluators_per_file, tag):
    FEATURE_TYPE = 'scalar'
    range = {
        'PitchCount':(0, 120),
        'PitchRange':(0, 100),
        'AveragePitchInterval':(0,40)
    }

    data_points = defaultdict(list)
    
    for file, evaluators in evaluators_per_file.items():
        for name, evaluator in evaluators.items():
            if evaluator.feature_type == FEATURE_TYPE:
                data_points[name].append(evaluator.feature)

    for name in data_points:
        kde = gaussian_kde(data_points[name])

        min_val, max_val = range[name]
        x = np.linspace(min_val, max_val, 100)
        pdf = kde.evaluate(x)
        plt.clf()
        plt.plot(x, pdf)
        figure_name = 'plots/pdf-{}-{}.png'.format(name, tag)
        np.save('plots/pdf-{}-{}'.format(name, tag), pdf)
        print('Saving figure to {}'.format(figure_name))
        plt.savefig(figure_name)

def process_histogram_features(evaluators_per_file, tag):
    FEATURE_TYPE = 'histogram'

    histograms = defaultdict(lambda : defaultdict(int))

    for file, evaluators in evaluators_per_file.items():
        for name, evaluator in evaluators.items():
            if evaluator.feature_type == FEATURE_TYPE:
                for key in evaluator.feature:
                    # print(key, evaluator.feature[key])
                    histograms[name][key] += evaluator.feature[key]

    for name in histograms:
        print("name: {}".format(histograms[name]))

    # Create pitch class plot
    plt.clf()
    pitch_class = np.zeros(12)
    for pitch, value in histograms["PitchClassHistogram"].items():
        pitch_class[util.NAMES[pitch]] = value
    plt.plot(util.PITCHES, pitch_class)
    np.save('plots/pdf-PitchClassHistogram-{}'.format(tag), pitch_class)
    plt.savefig("plots/pdf-PitchClassHistogram-{}.png".format(tag))

    # Create pitch-transition matrix plot
    pitch_transition_heatmap = np.zeros((12,12))
    for pitch_pair, value in histograms["PitchClassTransitionMatrix"].items():
        y = util.NAMES[pitch_pair[0]]
        x = util.NAMES[pitch_pair[1]]
        pitch_transition_heatmap[x][y] = value

    plt.clf()
    sns.heatmap(pitch_transition_heatmap, xticklabels=util.PITCHES, yticklabels=util.PITCHES)
    np.save('plots/pdf-PitchClassTransitionMatrix-{}'.format(tag), pitch_transition_heatmap)
    plt.savefig("plots/pdf-PitchClassTransitionMatrix-{}.png".format(tag))

def compute_intra(evaluators_per_file, tag):

    process_scalar_features(evaluators_per_file, tag)
    process_histogram_features(evaluators_per_file, tag)
    
    
if __name__=="__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('midi')
    parser.add_argument('--tag', default='')
    ARGS = parser.parse_args()

    if os.path.isfile(ARGS.midi):
        stream = music21.converter.parse(ARGS.midi)
        evaluate(stream)
    else:
        evaluators_per_file = {}
        path = ARGS.midi
        for root, dirs, files in os.walk(path):
            for file in files:
                if not file.startswith('.') \
                    and (file.endswith('.mid') or file.endswith('.midi')) \
                    and not 'seed' in file:
                    print('Processing {}'.format(file))
                    try:
                        stream = music21.converter.parse(os.path.join(root, file))
                        evaluators_per_file[file] = evaluate(stream)
                    except Exception as e:
                        print("Skipping {}".format(file))
        
        compute_intra(evaluators_per_file, ARGS.tag)