###------------------------------------------------------###
###          Replay and Remember Memory Class            ###
###------------------------------------------------------###

import numpy as np

from hyperparameters import *

class ReplayMemory:
    def __init__(self, memory_size, state_size, action_size):

        # set the state size, HEIGHT : default 84px
        self.state_height = state_size[0]

        # set the state size, WIDTH : default 84px
        self.state_width = state_size[1]

        # set the state size, DEPTH : default 1, grayscale
        self.state_depth = state_size[2]

        # set the action size, 4 actions
        self.action_size = action_size

        # initial size
        self.size = 0

        # set the max size of the remember and replay memory
        self.maxsize = memory_size

        # default current index
        self.current_index = 0

        # create the current state of the game (1,000,000, 64)
        self.current_state = np.zeros([memory_size, self.state_height, self.state_width, self.state_depth])

        # create the next state of the game (1,000,000, 64)
        self.next_state = np.zeros([memory_size, self.state_height, self.state_width, self.state_depth])

        # reward array (1,000,000)
        self.reward = np.zeros([memory_size])

        # integer action
        self.action = [0]*memory_size 

        # Boolean (terminal transition?)
        self.done = [False]*memory_size 

    def remember(self, current_state, action, reward, next_state, done):
        # Stores a single memory item
        self.current_state[self.current_index,:] = current_state
        self.action[self.current_index] = action
        self.reward[self.current_index] = reward
        self.next_state[self.current_index,:] = next_state
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
            current_sample = np.random.choice(self.size,sample_size,replace=False)
            
            # Slice memory into training sample
            current_state = self.current_state[current_sample,:]
            action = [self.action[j] for j in current_sample]
            reward = self.reward[current_sample]
            next_state = self.next_state[current_sample,:]
            done = [self.done[j] for j in current_sample]
            
            # Obtain model's current Q-values
            model_targets = model.predict(current_state)
            
            # Create targets from argmax(Q(s+1,a+1))
            # Use the target model!
            targets = reward + gamma*np.amax(target_model.predict(next_state),axis=1)
            # Absorb the reward on terminal state-action transitions
            targets[done] = reward[done]
            # Update just the relevant parts of the model_target vector...
            model_targets[range(sample_size),action] = targets
            
            # Update the weights accordingly
            model.fit(current_state,model_targets,
                     epochs=1,verbose=show_fit,batch_size=sample_size)
            
        # Once we have finished training, update the target model
        target_model.set_weights(model.get_weights())
