import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from scipy.stats import entropy

def compare_scalar_feature(features, feature_name):
    pairs = [
        ('batch-generated-bach', 'bach'),
        ('baseline-bach', 'bach'),
        ('rl-bach', 'bach'),
        ('chopin', 'bach'),
        ('batch-generated-chopin', 'chopin'),
        ('baseline-chopin', 'chopin'),
        ('bach', 'chopin'),
    ]

    eps = 1e-16

    for tag1, tag2 in pairs:
        if tag1 != tag2:
            kl = entropy(
                features[tag1][feature_name]+eps,
                features[tag2][feature_name]+eps)
            print('{}: {} vs. {}: {:.3f}'.format(
                feature_name,
                tag1, tag2, kl))

def plot_scalar_feature(features, feature_name):
    labels = {
        'bach':'Bach',
        'chopin':'Chopin',
        'batch-generated-bach':'Bach (LSTM)',
        'batch-generated-chopin':'Chopin (LSTM)',
        'baseline-bach':'Bach (Baseline)',
        'baseline-chopin':'Chopin (Baseline)',
        'rl-bach':'Bach (RL)'
    }

    styles = {
        'bach':'b',
        'baseline-bach':'b:',
        'batch-generated-bach':'b--',
        'rl-bach':'g',
        'chopin':'r',
        'baseline-chopin':'r:',
        'batch-generated-chopin':'r--'
    }

    tags = [
        'bach', 'baseline-bach', 'batch-generated-bach', 'rl-bach', 
        'chopin', 'baseline-chopin', 'batch-generated-chopin']

    plt.clf()
    for tag in tags:
        normalized = features[tag][feature_name] / sum(features[tag][feature_name])
        plt.plot(normalized, styles[tag], label=labels[tag])

    plt.title("Distribution of {}".format(feature_name))
    plt.legend()
    
    plt.grid()

    if feature_name == "PitchRange":
        plt.xlabel("Pitch range in semitones (= highest note - lowest note)")
    elif feature_name == "PitchCount":
        plt.xlabel("Number of distinct pitches")
    elif feature_name == "AveragePitchInterval":
        plt.xlabel("Average interval between consecutive notes")

    plt.tight_layout()
    plt.savefig('plots/overall-{}.png'.format(feature_name))

if __name__=="__main__":
    path = './plots'

    features = defaultdict(dict)

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.npy'):
                feature = file.split('-')[1]
                tag = '-'.join(file.split('-')[2:]).replace('.npy','')

                features[tag][feature] = np.load(os.path.join(
                    root, file
                ))

                print(tag, feature)

    # Compare scalar features
    compare_scalar_feature(features, 'PitchClassHistogram')
    compare_scalar_feature(features, 'AveragePitchInterval')
    compare_scalar_feature(features, 'PitchCount')
    compare_scalar_feature(features, 'PitchRange')

    # Plot scalar features
    plot_scalar_feature(features, 'PitchClassHistogram')
    plot_scalar_feature(features, 'AveragePitchInterval')
    plot_scalar_feature(features, 'PitchCount')
    plot_scalar_feature(features, 'PitchRange')

    # for tag in features:
        # for name in features[tag]:
            # print(features[tag][name])
