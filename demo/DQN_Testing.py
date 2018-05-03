# ## Project Demo
# Just run through these to watch breakout run! We handle processing the frame input and rewards. We then make the environment and model then run the demo.


# #### Imports as needed:
from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

import matplotlib.pyplot as plt
from IPython.display import display
from IPython.display import clear_output

from time import sleep


# #### Global Variables:
# height and width
INPUT_SHAPE = (84, 84)
# frames used together to input (channels) into the convolutional model
WINDOW_LENGTH = 4


# #### Atari Processor class for processing observations and rewards:
class AtariProcessor():
    """
    Atari Processor for processing
    """
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch


def demo():
    seeds = [2095, 274, 1770, 263, 1115, 403]

    for i in range(len(seeds)):
        lives = 5
        env.seed(seeds[i])
        # initialize a frame set to 0s
        frames = np.zeros((1,WINDOW_LENGTH,)+INPUT_SHAPE)

        # reset the observation
        observation = env.reset()

        # process the first observation as an initial frame set
        myframe = processor.process_state_batch(processor.process_observation(observation))
        for i in range(WINDOW_LENGTH):
            frames[:,i,:,:] = myframe

        # initializers
        done = False
        while not done:
            env.render()
            action = np.argmax(model.predict(frames))
            sleep(.04)

            modified_action = action+1
            observation,reward,done,info = env.step(modified_action)

            myframe = processor.process_state_batch(processor.process_observation(observation))

            # move the frame along
            frames[:,0:WINDOW_LENGTH-1,:,:] = frames[:,1:WINDOW_LENGTH,:,:]
            frames[:,WINDOW_LENGTH-1,:,:] = myframe

            if lives != info['ale.lives']:
                lives = info['ale.lives']
                observation,reward,done,info = env.step(1)
        sleep(3)


# #### Gym Environment set up:
env = gym.make('Breakout-v4')
processor = AtariProcessor()
nb_actions = env.action_space.n-1
print('Modified Number of Actions:', nb_actions)


# #### Building the Model:
# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model = Sequential()

if K.image_dim_ordering() == 'tf':
    # (width, height, channels)
    model.add(Permute((2, 3, 1), input_shape=input_shape))
elif K.image_dim_ordering() == 'th':
    # (channels, width, height)
    model.add(Permute((1, 2, 3), input_shape=input_shape))
else:
    raise RuntimeError('Unknown image_dim_ordering.')

model.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
model.compile(loss='mse',optimizer=Adam(lr=0.00025))

# #### Loading the weights that you want!
weights_filename = 'breakout-v4-weights-18-04-27-18-28.h5'
model.load_weights(weights_filename)

# render initial environment window
env.render()
demo()
