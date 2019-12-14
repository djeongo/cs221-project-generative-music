import gym
import logging

from gym import error, spaces, utils
from gym.utils import seeding
from util import note_to_number

import numpy as np
import random

logger = logging.getLogger(__name__)

C_MAJOR = ['C','D','E','F','G','A','B']
TONIC = 'C'

class MelodyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_len, num_classes, reward_network, index_to_word):
        self.max_len = max_len
        self.num_classes = num_classes
        self.state = [np.random.randint(0, num_classes)]
        self.action_space = spaces.Discrete(num_classes)
        self.actions = []
        self.index_to_word = index_to_word
        self.word_to_index = {}
        for index, word in enumerate(self.index_to_word):
            self.word_to_index [word] = index
        self.reward_network = reward_network

    def _is_note(self, action):
        return not self.index_to_word[action].startswith('w')
        
    def _reward_wrong_key(self, new_action):
        reward = -1
        if self._is_note(new_action):
            note_name = self.index_to_word[new_action]
            if note_name[:-1] in C_MAJOR:
                reward = 1
        logger.info('Computed wrong_key reward: {}'.format(reward))
        return reward

    def _reward_repeat_notes(self, new_action):
        # Penalize if the same not repeats 4 times
        NUM_REPEATS = 4
        PENALTY = -1

        reward = 0
        # search last four notes
        num_notes = 0
        num_same_notes = 0
        for action in self.actions[::-1]:
            if self._is_note(action):
                num_notes += 1
                if new_action == action:
                    num_same_notes += 1
            if num_notes == NUM_REPEATS - 1:
                break

        if num_same_notes == NUM_REPEATS - 1:
            reward = PENALTY

        logger.info("Computing repeat_notes reward: {}".format(reward))
        return reward

    def _reward_intervals(self, new_action):
        OCTAVE = 12
        reward = 0
        if self._is_note(new_action):
            n1 = note_to_number(self.index_to_word[new_action])
            for action in self.actions[::-1]:
                if self._is_note(action):
                    n2 = note_to_number(self.index_to_word[action])
                    interval = abs(n2-n1)
                    if interval > OCTAVE:
                        reward = -1
                    break
        logger.info("Computing intervals reward: {}".format(reward))
        return reward

    def _reward_lstm(self, action):
        state = self.state
        # logger.info("state: {}".format(state))
        action_probs = self.reward_network.predict(np.array([self.state])).flatten()
        # print(action_probs)
        reward=action_probs[action]
        logger.info("Computing LSTM reward: {}".format(reward))
        return reward

    def _reward_end_in_tonic(self, action, done):
        reward = 0
        note_name = self.index_to_word[action][:-1]
        if done and note_name == TONIC:
            reward = 10
        logger.info("Computing end-in-tonic reward: {}, note_name: {}, done: {}".format(
            reward,
            note_name,
            done))
        return reward

    def _compute_reward(self, action, done):
        # Note in the wrong key given low reward
        # Melody starting and finishing with the same tonic note of the key should be given high reward
        #   How to define start and end of melody?
        # Repetition of the same note should be minimized
        #   A single note should not be repeated more than four times in a row
        # If a given sequence of notes are too similar with itself from previous time steps, low reward
        # Notes follow certain intervals, large jumps greater than an ocatve should receive negative reward
        reward = 0
        reward += self._reward_wrong_key(action)
        reward += self._reward_repeat_notes(action)
        reward += self._reward_intervals(action)
        reward += self._reward_lstm(action)
        reward += self._reward_end_in_tonic(action, done)
        return reward

    def step(self, action):
        # one of the possible notes or no action
        # TODO: Need to represent stop action since we don't have velocity any more?
        done = False
        done = len(self.actions) == self.max_len
        reward = self._compute_reward(action, done)

        self.actions.append(action)
        self.state.append(action)
        
        return self.state, reward, done, {}

    def reset(self):
        # initialize self.state
        self.state = random.sample([
            self.word_to_index['C5'],
            self.word_to_index['D5'],
            self.word_to_index['E5'],
            self.word_to_index['F5'],
            self.word_to_index['G5'],
            self.word_to_index['A5'],
            self.word_to_index['B5']], 1)
        self.actions = []

    def render(self, mode='human'):
        # probably leave it unimplemented or return piano roll
        print("render: {}".format(self.state))

    def close(self):
        # close viewer?
        pass 
