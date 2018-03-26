from collections import deque
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np
import random


# DQNAgent for breakout
class DQNAgent():
    def __init__(self, input_shape, action_space, model='Dense'):

        # input layer into our first dense layer
        self.input_shape = input_shape
        # with downsample: (210, 160, 3) -> (105, 80)
        #self.input_shape = 80

        # output layer mapped to an action
        self.action_space = action_space

        # hyper parameters for the DQN
        self.learning_rate = 0.001
        self.epsilon = 1.0              # exploration rate
        self.epsilon_min = 0.01         # minimum exploration rate
        self.epsilon_decay = 0.999      # decay rate for exploration
        self.batch_size = 32            # batches
        self.gamma = 0.95               # discount rate


        self.memory_size = 1000000      # size of the deque

        # create replay memory using deque
        self.memory = deque(maxlen=self.memory_size)

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
        #model.add(keras.layers.Conv2D(16, kernel_size = (4,4),
        #                                  activation ='relu',
        #                                  input_shape = self.input_shape))

        # with downsample (105, 80, 1)

        print(self.input_shape)
        model.add(keras.layers.Conv2D(16, kernel_size = (4,4),
                                          activation ='relu',
                                          input_shape = (84, 84, 1) ))

        #model.add(keras.layers.Flatten())

          
        # fed into a lower dimensional convolutional layer
        model.add(keras.layers.Conv2D(8, (8,8), activation ='relu'))

        # dense layer 64
        model.add(keras.layers.Dense(32, activation = 'relu'))


        # classify with softmax into a category
        model.add(keras.layers.Dense(self.action_space, activation = 'softmax'))

        # compile the model
        model.compile(loss = keras.losses.categorical_crossentropy,
                      optimizer = keras.optimizers.Adam(lr = self.learning_rate),
                      metrics = ['accuracy'])

        # show summary
        model.summary()

        # return the built and compiled model
        return model

    # remembers by appending to the memory deque
    # Input:  state, action, reward value, the next state, and done flag
    # Output: None, but appends to the memory deque
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # acts upon the game given a state
    # Input:  state of the game
    # Output: returns the action taken
    def act(self, state):

        # with some probability from our epsilon annealing,
        if np.random.rand() <= self.epsilon:
            # select a random action
            return random.randrange(self.action_space)   # returns action

        # otherwise,
        else:
            # act given a prediction
            state = state.reshape(1,84,84,1)
            act_values = self.model.predict(state)

            decision = np.argmax(act_values[0])
            if decision > 3:
                print('Decision:', decision, 'supposed to be 0-3!')
                decision = 0

            return decision          # returns action

    # replays the past frames labeled by the batch size
    # Input:  Batch size of the most recent frames
    # Output: None, but decays the epsilon learning if the minimum has been exceeded
    def replay(self, batch_size):

        # sets the minibatch from a random sample from the deque and batch size
        minibatch = random.sample(self.memory, batch_size)

        # iterate over the minibatch and collect the reward as a target
        for state, action, reward, next_state, done in minibatch:
            target = reward

            #print(next_state.shape)
            #print(next_state)

            next_state = next_state.reshape(1,84,84,1)

            # if we're not done
            if not done:
                # set a new target dependant on the reward
                # added from gamma * the max of the prediction of what might be in the next state
                # Q-values for each action
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))


            state = state.reshape(1,84,84,1)

            # set the finished target as a prediction of the current state
            target_f = self.model.predict(state)

            # set it as the target collected
            target_f[0][action] = target

            # fit our model with the a new target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # if we're above the epsilon minimum,
        if self.epsilon > self.epsilon_min:
            # decay our learning from our decay constant
            self.epsilon *= self.epsilon_decay
   

    # hard exits the game
    # Input: None
    # Ouput: None, but saves and exits the game
    def quit(self):
        
        # set the file name
        fn = 'weights/breakout-v4-ram-weights-' +                       \
        str(datetime.datetime.now().strftime("%y-%m-%d-%H-%M")) + '.h5'

        # save the model
        self.agent.save(fn)

        # exit
        print('Exiting..')
        pg.quit()
        sys.exit()

    # load the weights for the game from previous runs
    # Input: filename input
    # Output: None
    def load(self, name):
        print('Loading weights from: ', name)
        self.model.load_weights(name)

    # saves the weights into a folder in ./weights/
    # Input:  filename 
    # Output: None, saves the file into a folder
    def save(self, name):
        print('Saving weights as: ', name)
        self.model.save_weights(name)

