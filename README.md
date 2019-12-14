# cs221-project-generative-music

# 
| file                         | description                                                         |   
| -----------------------------|:------------------------------------------------------------------- |
| src/midi_to_words_music21.py | utility for encoding MIDI files into notes and rests representation |
| src/encoding_to_wordvec.py   | utility for training a Word2Vec model from encoded representation   |
| src/baseline.py              | script to train baseline model                                      |
| src/train.py                 | script to train LSTM model using Keras                              |
| src/test_word2vec.py         | script to generate music using LSTM model                           |
| src/util.py                  | common utility functions                                            |
| src/gym-melody/gym_melody/envs/melody_env.py | OpenAI Gym environment for training RL model        |

# Preprocessing
* Encode MIDI files for training
    ```
    $ python src/midi_to_words_music21.py data/bach data/bach-encoded
    ```
where `data/bach` has input MIDI files and `data/bach-encoded` is output directory where encoded
training data get generated.

* Train Word2Vec model
    ```
    $ python src/encoding_to_wordvec.py data/bach-encoded
    ```
uses encoded data in `data/bach-encoded` to train word2vec model.
Generates `word2vec.model` and a pickled file `wv.pickle` containig only the
word2vec vectors.

# Training Baseline
* Train baseline model
    ```
    $ python src/baseline.py --train-data data/bach-encoded --output-path data/bach-baseline
    ```
This script also generates baseline music to `data/bach-baseline`

# Training LSTM
* Train LSTM model
    ```
    $ python src/train.py --lstm-size 256 --train-data data/bach-encoded/ --max-len 50 --dropout 0.2 --lstm-layers 2 --batch-size 8192 --wv-file wv.pickle
    ```
Trains the model with two-layer LSTM of size 256 and dropout 0.2 using batch-size 8192.

# Training Reinforcement Learning Model
* Train RL model
