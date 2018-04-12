###------------------------------------------------------###
###          Replay and Remember Memory Class            ###
###------------------------------------------------------###

import numpy as np

from hyperparameters import *

def normalize_states(current_frame_history):
    # expand dimensions to (1, 84, 84, 5) from (84, 84, 5)
    # normalize 0-255 -> 0-1 to reduce exploding gradient
    return np.dtype(float).type(current_frame_history) / 255.

class ReplayMemory:
    def __init__(self, memory_size, state_size, action_size):

        # set the state size, HEIGHT : default 84px
        self.state_height = state_size[0]

        # set the state size, WIDTH : default 84px
        self.state_width = state_size[1]

        # set the state size, DEPTH : default 4, for 4 frames
        # we then extend it to make room for the current and next frame
        # current frame: [0, 1, 2, 3]
        # next frame: [1, 2, 3, 4]
        self.state_depth = state_size[2] + 1 

        # set the action size, 4 actions
        self.action_size = action_size

        # initial size
        self.size = 0

        # set the max size of the remember and replay memory
        self.maxsize = memory_size

        # default current index
        self.current_index = 0

        # create the current state of the game (1,000,000, 64)
        self.states = np.zeros([memory_size, self.state_height, self.state_width, self.state_depth])

        # reward array (1,000,000)
        self.reward = np.zeros([memory_size])

        # integer action
        self.action = [0]*memory_size 

        # Boolean (terminal transition?)
        self.done = [False]*memory_size 

    def remember(self, current_state, action, reward, next_state, done):

        # create an extended frame state
        frame_state = np.zeros([84, 84, 5], dtype=np.uint8)

        # append the current state, 0, 1, 2, 3
        frame_state[:, :, 0:4] = current_state
        # get the last state, 4
        frame_state[:, :, 4] = next_state[:, :, 3]


        # Stores a single memory item
        self.states[self.current_index,:] = frame_state

        self.action[self.current_index] = action
        self.reward[self.current_index] = reward
        self.done[self.current_index] = done
        self.current_index = (self.current_index+1)%self.maxsize
        self.size = max(self.current_index,self.size)
    
    def replay(self, model, target_model):
        # Run replay!
        
        # set the number of samples to train on
        num_samples = hp['REPLAY_ITERATIONS']

        # set the sample size out of the memory bank 
        sample_size = hp['REPLAY_SAMPLE_SIZE']

        # discount rate
        gamma = hp['GAMMA']

        # show the learning fit
        show_fit = hp['SHOW_FIT']

        # Can't train if we don't yet have enough samples to begin with...
        if self.size < sample_size:
            return
                
        for i in range(num_samples):
                            
            # Select sample_size memory indices from the whole set
            current_sample = np.random.choice(self.size, sample_size, replace=False)
            
            # Slice memory into training sample
            # current state is frames [0, 1, 2, 3]
            # and normalize states [0,1] instead of 0-255
            current_state = normalize_states(self.states[current_sample, :, :, 0:4])

            # next_state is frames [1, 2, 3, 4]
            # and normalize states [0,1] instead of 0-255
            next_state = normalize_states(self.states[current_sample, :, :, 1:5])


            action = [self.action[j] for j in current_sample]
            reward = self.reward[current_sample]
            done = [self.done[j] for j in current_sample]
            
            # Obtain model's current Q-values
            model_targets = model.predict(current_state)
            
            # Create targets from argmax(Q(s+1,a+1))
            # Use the target model!
            targets = reward + gamma * np.amax(target_model.predict(next_state),axis=1)
            # Absorb the reward on terminal state-action transitions
            targets[done] = reward[done]
            # Update just the relevant parts of the model_target vector...
            model_targets[range(sample_size),action] = targets
            
            # Update the weights accordingly
            model.fit(current_state, model_targets,
                     epochs=1,verbose=show_fit,batch_size=sample_size)
            
 
    def target_update(self, model, target_model):
        print('Updating target model weights from model weights')
        # target model weights <- model weights
        target_model.set_weights(model.get_weights())
