###-----------------------------------------###
###          Deep Q Agent Class             ###
###-----------------------------------------###

from collections import deque
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model

from hyperparameters import *
import numpy as np
import random

import sys
import pickle
import datetime

import sys

def find_action(action):
    # actions:
    # 0: no-op 1: fire 2: right 3: left
    if action is 0:
        return 'no-op'
    elif action is 1:
        return 'fire'
    elif action is 2:
        return 'move right'
    elif action is 3:
        return 'move left'

# DQNAgent for breakout
class DQNAgent():
    def __init__(self, input_shape, action_space, model='Dense'):

        # input layer into our first dense layer
        # with downsample: (210, 160, 3) -> (105, 80)
        self.input_shape = input_shape

        # output layer mapped to an action
        self.action_space = action_space     


        if model is 'Dense':
            # build the dense model model 
            self.model = self.build_dense_model()

        elif model is 'Convolutional':
            # build the convolutional model 
            self.model = self.build_convolutional_model()

        #self.model.load_weights('pong_weights-18-01-28-16-18.h5')

    # build the model
    # Input:  none
    # Output: Returns the built and compiled model
    def build_dense_model(self):

        # create a sequential, dqn model
        model = Sequential()

        # 24 layer dense relu, 128 input dimension
        model.add(Dense(48, input_dim=1, activation='relu'))

        # another 24 dense relu
        model.add(Dense(24, activation='relu'))

        # a final linear activation for a certain action (move left, move right, noop, fire)
        model.add(Dense(self.action_space, activation='linear'))

        # compile the model
        # try mean squared error or logcosh, clog of hyperbolic cosine
        model.compile(loss = 'mse',
                      optimizer = keras.optimizers.Adam(lr=self.learning_rate),
                      metrics = ['accuracy'])

        # show summary
        model.summary()

        # return the built and compiled model
        return model


    def build_convolutional_model(self):

        # create a sequential, dqn model
        model = Sequential()

        # without downsample: (210, 160, 3)
        # with downsample: (84, 84, 1)
        model.add(keras.layers.Conv2D(32, kernel_size = (4,4),
                                          activation ='relu',
                                          input_shape = self.input_shape ))

        #model.add(keras.layers.Flatten())

        # fed into a lower dimensional convolutional layer
        model.add(keras.layers.Conv2D(64, (2,2), activation ='relu'))

        # fed into a lower dimensional convolutional layer
        model.add(keras.layers.Conv2D(64, (1,1), activation ='relu'))

        model.add(keras.layers.Flatten())

        # dense layer 64
        model.add(keras.layers.Dense(512, activation = 'relu'))

        # classify with softmax into a category 
        model.add(keras.layers.Dense(self.action_space, activation = 'linear'))

        # try mse, mean squared error or logcosh, log of hyperbolic cosine
        model.compile(loss = keras.losses.logcosh if hp['LOSS'] is 'logcosh' 
                        else keras.lossses.mse    if hp['LOSS'] is 'mse'
                        else keras.losses.logcosh,
                      optimizer = keras.optimizers.Adam(lr = hp['LEARNING_RATE']),
                      metrics = ['accuracy'])

        # show summary
        model.summary()

        # return the built and compiled model
        return model

    # acts upon the game given a state
    # Input:  state of the game
    # Output: returns the action taken
    def act(self, Q):

        # with some probability from our epsilon annealing,
        if np.random.rand() <= hp['EPSILON']:
            # select a random action
            rand = random.randrange(self.action_space)

            # print q and decision
            if hp['WATCH_Q']:
                print ('Random Action! Q:', Q, 'decision:', find_action(rand))

            return Q[0][rand], rand                  # returns action

        # otherwise,
        else:
            decision = np.argmax(Q)

            # print q and decision
            if hp['WATCH_Q']:
                print ('Q:', Q, 'decision:', find_action(decision))

            return Q[0][decision], decision          # returns action

    # hard exits the game
    # Input: None
    # Ouput: None, but saves and exits the game
    def quit(self, mean_times):

        # save the model
        self.save()

        print('Saving stats..')
        # saving stats
        with open('stats/mean_times.data', 'wb') as f:
            pickle.dump(mean_times, f)

        # exit
        print('Exiting..')
        sys.exit()

    # load the weights for the game from previous runs
    # Input: filename input
    # Output: None
    def load(self, name):
        name = 'weights/' + name 
        print('Loading weights from: ', name)
        self.model.load_weights(name)

    # saves the weights into a folder in ./weights/
    # Input:  filename 
    # Output: None, saves the file into a folder
    def save(self):
        # set the file name
        fn = 'weights/breakout-v4-weights-' + \
        str(datetime.datetime.now().strftime("%y-%m-%d-%H-%M")) + '.h5'

        print('Saving weights as: ', fn)
        self.model.save_weights(fn)

