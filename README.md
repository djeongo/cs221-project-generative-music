# cs221-project-generative-music

# 
* src/midi_to_words_music21.py
** utility for encoding MIDI files into notes and rests representation
* src/encoding_to_wordvec.py
** utility for training a Word2Vec model from encoded representation
* src/baseline.py
** script to train baseline model
* src/train.py
** script to train LSTM model using Keras
* src/util.py
** common utility functions

# Preprocessing
1. Encode MIDI files for training
    ```
    $ python src/midi_to_words_music21.py data/bach data/bach-encoded
    ```
where data/bach has MIDI files and data/bach-encoded is where encoded
training data get generated. 

1. Train Word2Vec model
    ```
    $ python src/encoding_to_wordvec.py data/bach-encoded
    ```
uses encoded data in data/bach-encoded to train word2vec model.
Generates word2vec.model and a pickled file wv.pickle containig only the
word2vec vectors.

# Training Baseline
1. 

# Training LSTM
1. Train LSTM model

# Training Reinforcement Learning Model
1. Train RL model
