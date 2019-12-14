import argparse
import collections
import gym
import logging
import numpy as np
import random
from keras.models import load_model
from keras.preprocessing import sequence
from train import build_model, get_embedding_matrix
from config import LSTM_UNITS, MAX_LEN, LOSS
from test_word2vec import notes_to_midi
import os

NUM_EPOCHS = 500
M = 100 # Num episodes
T = 200 # Timesteps
MINI_BATCH_SIZE=64

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if len(logger.handlers) == 0:
    logger.addHandler(logging.StreamHandler())

def round_list(l):
    return ' '.join(['{:.2f}'.format(x) for x in l])

def load_lstm(lstm_model):
    model = load_model(lstm_model)
    model.summary()
    return model

class MelodyRL:
    def __init__(self, lstm_model, word2vec_model):
        reward_network = load_lstm(lstm_model)
        embedding_matrix, word_to_index, index_to_word = get_embedding_matrix(
            word2vec_model
        )
        self.index_to_word = index_to_word
        self.num_classes = len(word_to_index)
        self.env = gym.make('gym_melody:melody-v0',
                            max_len=MAX_LEN,
                            num_classes=self.num_classes,
                            reward_network=reward_network,
                            index_to_word=index_to_word)
        self.q_network = build_model(
            embedding_matrix,
            lstm_layers=2,
            lstm_size=256,
            dropout=0.5,
            activation='linear')
        self.q_network.load_weights(lstm_model)
        self.gamma = 0.99

    def sample(self, eps, x):
        if np.random.random() < eps or len(x) == 0:
            action = np.random.randint(self.num_classes)
            logger.info('Taking random action: {}'.format(action))
        else:
            # choose action based on Q network
            # logger.info('x: {}'.format(x))
            y = self.q_network.predict(np.array([x])).flatten()
            print(y)
#            print('q-network: {}'.format(y))
            action = np.argmax(y)
            logger.info('Taking q-network action: {}'.format(action))
        return action

    def build_train_data(self, minibatch):
        # Build input data suitable for LSTM
        #print(len(minibatch))
        x = []
        y = []
        for i, sample in enumerate(minibatch):
            state, action, reward, next_state = sample
            # logger.info('sample: state: {} action: {} reward: {} next_state: {}'.format(
                # state, action, reward, next_state))
            q_value_next = self.q_network.predict(np.array([next_state]))
            q_value = q_value_next[0,:]
#            print('action:{} reward:{}'.format(action, reward))
            q_value[action] = reward + self.gamma*np.max(q_value_next)
#            print('q_value:{}'.format(q_value))
            # logger.info('{} state:{}'.format(i, state))
            x.append(state)
            y.append(q_value)

        # for state in x:
        #     print(len(state))
        # for q_value in y:
        #     print(len(q_value))
        # logger.info('x:{}'.format(x))
        # logger.info('y:{}'.format(y))
        return sequence.pad_sequences(x, maxlen=MAX_LEN), np.stack(y)

    def train(self):
        D = collections.deque(maxlen=500)
        rewards = []
        eps_schedule = np.linspace(0.5, 0.01, NUM_EPOCHS)
        epoch_rewards = []
        for epoch in range(NUM_EPOCHS):
            eps = eps_schedule[epoch]
            episode_rewards = []
            for episode in range(M):
                self.env.reset()
                episode_reward = 0
                done = False
                t = 0
                while not done:
                    print('epoch:{}/{} episode: {}/{} t: {}/{} episode_rewards: {} epoch_rewards:{}'.format(
                        epoch, NUM_EPOCHS, episode, M, t,T, round_list(episode_rewards), epoch_rewards))
                    state = self.env.state
                    action = self.sample(eps, state)
                    next_state, reward, done, info = self.env.step(action)
                    episode_reward += reward
                    # s.append(
                    # print('action: {}'.format(self.env.actions))
    #                print('state:\n{}'.format(state))
                    D.append((state.copy(), action, reward, next_state.copy()))
                    # logger.info("len(D): {}".format(len(D)))
                    minibatch = random.sample(D, min(len(D), MINI_BATCH_SIZE))
                    # logger.info("len(minibatch): {}".format(len(minibatch)))
                    # TODO: Should this be done more intermittently?
                    t+=1

                x,y = self.build_train_data(minibatch)
                print(y)
                loss = self.q_network.train_on_batch(x, y)
                print("loss: {}".format(loss))

                episode_rewards.append(episode_reward)
                print('rewards: {}'.format(episode_rewards))
                
            epoch_rewards.append(np.mean(episode_rewards))
            self.q_network.save('q-network-epoch-{}.h5'.format(epoch))

    def test(self, q_network_path, output_path, tag):
        self.q_network = load_model(q_network_path)
        self.env.reset()
        t = 0
        done = False
        eps = 0.01
        for i in range(500):
            t+=1
            state = self.env.state
            action = self.sample(eps, state)
            print('action: ', action)
            next_state, reward, done, info = self.env.step(action)
        
        print(state)
        notes_to_midi(
            state,
            self.index_to_word,
            os.path.join(output_path, 'generated-{}.mid'.format(tag)))



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lstm-model', required=True)
    parser.add_argument('--word2vec-model', required=False)
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--q-network')
    parser.add_argument('--output-path', default='/tmp/')
    parser.add_argument('--tag', default='')
    
    ARGS = parser.parse_args()
    instance = MelodyRL(
        ARGS.lstm_model,
        ARGS.word2vec_model)

    if not ARGS.test:
        instance.train()
    else:
        for i in range(50):
            instance.test(ARGS.q_network, ARGS.output_path, i)
