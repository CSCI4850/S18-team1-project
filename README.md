<h1>Neocognitron Group</hi>

## Breakout Atari Agent
[[https://github.com/CSCI4850/S18-team1-project/blob/master/breakout.png|alt=breakout]]

### Model:
Our model consists of a Convolutional Neural Network with a preprocessed frame from Breakout of a (210, 160, 3) tuple => (84, 84) grayscale down-sized frame and a linear output size of 4 (no-op, fire, move left, move right). The model uses the Adam optimizer with a logcosh loss function.

### Components:
1. breakout.py:
  The main breakout game loop. Integrates with DQNAgent.py and ReplayMemory.py.

2. DQNAgent.py
  The Deep Q Network Agent for learning the breakout game.

3. ReplayMemory.py
  The Remembering and Replaying for the DQNAgent to learn.
  
4. hyperparameters.py
  All of the hyperparameters
  
#### Breakout Main Loop: 
    1. Height, in pixels that the frame will be preprocessed to
    2. Width, in pixels that the frame will be preprocessed to
    3. Max Episodes, how many episodes to iterate through until the training is complete (death or game completeion)
    4. Target Update, when to update the target agent model
    
#### DQNAgent:
    1. Learning Rate, the learning rate of the Adam Optimizer
    2. Epsilon, the exploration vs exploitation rate: when to take a random action or an action that was learned
    3. Epsilon Decay, at what rate to decay or anneal the exploration rate: when to lower taking random actions
    4. Epsilon Minimum, at what minimum to take random actions

#### Replay and Remember Memory:
    1. Memory Size, how big the memory bank will be
    2. Gamma, integration of rewards, discount factor, preference for present rewards as opposed to future rewards
    3. Replay Iterations, how many total iterations or samples from history we will replay from
    4. Replay Sample Size, how many from the memory bank of those samples we will actually replay
    
### Instructions:
To start the breakout game with the DQN Agent, run ```python3 breakout.py```
<br>
To change how the DQN Agent learns, modify hyperparameters.py

