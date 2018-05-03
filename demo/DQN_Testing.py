
# coding: utf-8

# ## Project Demo
# Just run through these to watch breakout run! We handle processing the frame input and rewards. We then make the environment and model then run the demo.

# #### Imports as needed:

# In[1]:


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

# In[2]:


# height and width
INPUT_SHAPE = (84, 84)
# frames used together to input (channels) into the convolutional model
WINDOW_LENGTH = 4


# #### Atari Processor class for processing observations and rewards:

# In[3]:


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

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


# #### Gym Environment set up:

# In[4]:


env = gym.make('BreakoutDeterministic-v4')
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
print(model.summary())


# #### Loading the weights that you want!

# In[6]:


weights_filename = 'breakout-v4-weights-18-04-27-18-28.h5'
model.load_weights(weights_filename)


# initialize a frame set to 0s
frames = np.zeros((1,WINDOW_LENGTH,)+INPUT_SHAPE)

# reset the observation
observation = env.reset()

# process the first observation as an initial frame set
myframe = processor.process_state_batch(processor.process_observation(observation))
for i in range(WINDOW_LENGTH):
    frames[:,i,:,:] = myframe

env.render()

# initializers
done = False
iteration = 0
    # modify the action space by adding one

    # process the frame

for i_episode in range(20):
    observation = env.reset()
    while not done:
        env.render()
        action = np.argmax(model.predict(frames))
        env.render()
        sleep(.1)

        modified_action = action+1
        observation,reward,done,_ = env.step(modified_action)

        myframe = processor.process_state_batch(processor.process_observation(observation))

        # move the frame along
        frames[:,0:WINDOW_LENGTH-1,:,:] = frames[:,1:WINDOW_LENGTH,:,:]
        frames[:,WINDOW_LENGTH-1,:,:] = myframe
        observation, reward, done, info = env.step(action)
        iteration += 1


# clear_output(wait=True)
