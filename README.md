<h1>Neocognitron Group</hi>

## Breakout Atari Agent
<img src="https://github.com/CSCI4850/S18-team1-project/blob/master/breakout.gif" width="200px" height="auto">
Results of training on a GTX 1080 for 10 hours.

### Model:
Our model consists of a Convolutional Neural Network with a preprocessed frame from Breakout of a (210, 160, 3) tuple => (84, 84) grayscale down-sized frame and a linear output size of 4 (no-op, fire, move left, move right) which gets reduced down to 3 (no-op, move left, move right) because (fire) in breakout is basically a no-op. The model uses the Adam optimizer with a logcosh, mean squared error, or huber loss function.

#### Requirements:
<b>python3</b>: need help installing? <a href="http://docs.python-guide.org/en/latest/starting/installation/">click here!</a><br>
<b>pip</b> (python's package manager): need help installing? <a href="https://www.makeuseof.com/tag/install-pip-for-python/">click here!</a><br>

    numpy==1.13.3
    scikit_image==0.13.1
    Keras==2.1.3
    gym==0.9.5
    gym[atari]
    h5py==2.7.1
    Pillow==5.1.0

or just do:
```pip install -r requirements.txt```<br>
<br>You will also need ```pip install tensorflow-gpu==1.7.0``` if you are using a GPU to train.<br>

### Python Components (located in src):
1. <b>breakout.py</b>:
  The main breakout game loop. Integrates with all of the components.

2. <b>DQNAgent.py</b>:
  The Deep Q Network Agent for learning the breakout game.

3. <b>ReplayMemory.py</b>:
  The Remembering and Replaying for the DQNAgent to learn.
  
4. <b>hyperparameters.py</b>:
  All of the hyperparameters
  
5. <b>discrete_frames.py</b>:
    Discrete frames into the model and memory. More memory footprint, more backpropogation steps.

6. <b>sliding_frames.py</b>:
    Sliding frames into the model and memory. Less memory footprint, less backpropagation steps.
    
7. <b>utils.py</b>:
    List of utility functions used by numerous components.
#### Breakout Main Loop: 
    'GAME' : 'BreakoutDeterministic-v4', # Name of which game to use
                                         # v1-4 Deterministic or Not

    'DISCRETE_FRAMING' : True,     # 2 discrete sets of frames stored in memory
    
    'LOAD_WEIGHTS' : '',           # Loads weights into the model if so desired
                                   # leave '' if starting from a new model

    'RENDER_ENV' : False,          # shows the screen of the game as it learns
                                   # massivly slows the training down when True
                                   # default: False

    'HEIGHT' : 84,                 # height in pixels
    'WIDTH'  : 84,                 # and width in pixels that the game window will get downscaled to
                                   # defaults: 84, 84

    'FRAME_SKIP_SIZE' : 4,         # how many frames we skip and and how many times we 
                                   # choose an action consecutively for that many frames.
                                   # default: 4
    
    'MAX_EPISODES' : 12000,        # defined as how many cycles of full life to end life or
                                   # winning a round
                                   # default: 12,000

    'MAX_FRAMES' : 50000000,       # max number of frames allowed to pass before stopping
                                   # default: 50,000,000 (how many google used)

    'SAVE_MODEL' : 500,            # how many episodes should we go through until we save the model?
                                   # default: whenever you want to save the model

    'TARGET_UPDATE' : 10000,       # on what mod epochs should we update the target network?
                                   # default: 10000
    
#### DQNAgent:
    'WATCH_Q' : False,             # watch the Q function and see what decision it picks
                                   # cool to watch
                                   # default: False

    'LEARNING_RATE' : 0.00025,     # learning rate of the Adam optimizer
                                   # default: 0.00025
        
    'INIT_EXPLORATION' : 1.0,      # exploration rate, start at 100%
    'EXPLORATION' : 1000000,       # how many frames we decay till
    'MIN_EXPLORATION' : 0.1,       # ending exploration rate
                                   # defaults: 1.0, 1,000,000, 0.1
    
    'OPTIMIZER' : 'Adam',          # optimizer used
                                   # default: RMSprop or Adam
    
    'MIN_SQUARED_GRADIENT' : 0.01, # epsilon rate
                                   # default: 0.01
    
    'GRADIENT_MOMENTUM' : 0.95,    # momentum into the gradient used
                                   # default: 0.95

    'LOSS' : 'huber',              # can be 'logcosh' for logarithm of hyperbolic cosine
                                   # or 'mse' for mean squared error
                                   # or 'huber' for huber loss
                                   # default: logcosh, mse, or huber
        
    'NO-OP_MAX' : 30,              # how many times no-op can be called at the beginning
                                   # of a single episode, reduces using the same state
                                   # at the beginning and increases variance of similar states
                                   # default: 30 (don't set this too high or we may lose before acting!)
#### Replay and Remember Memory:
    'SHOW_FIT' : 0,                # shows the fit of the model and it's work, turn to 0 for off
                                   # default: 0 for off
    
    'REPLAY_START' : 50000,        # when to start using replay to update the model
                                   # default: 50000 frames

    'MEMORY_SIZE' : 1000000,       # size of the memory bank
                                   # default: 1,000,000

    'GAMMA' : 0.99,                # integration of rewards, discount factor, 
                                   # preference for present rewards as opposed to future rewards
                                   # default: 0.99
    # 4 * 8 = 32 batch
    'REPLAY_ITERATIONS' : 4,       # how many irerations of replay
                                   # default: 4

    'BATCH_SIZE' : 8               # batch size used to learn
                                   # default: 8
                                   
### Instructions:
To start the breakout game with the DQN Agent, run ```python3 breakout.py```
<br>
To change how the DQN Agent learns, modify hyperparameters.py

### Demo (located in demo):
To start the demo, run ```python3 DQN_Testing.py```<br>
Alternatively, there is a python notebook under DQN_Testing.ipynb which renders every 6 frames.
<br>

### References:
1. http://docs.python-guide.org/en/latest/starting/installation/
2. https://www.makeuseof.com/tag/install-pip-for-python/
3. https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
4. https://github.com/dennybritz/reinforcement-learning/issues/30
5. https://github.com/tokb23/dqn/blob/master/dqn.py
6. https://github.com/jcwleo/Reinforcement_Learning/blob/master/Breakout/Breakout_DQN_class.py
7. https://medium.com/mlreview/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55
8. https://becominghuman.ai/beat-atari-with-deep-reinforcement-learning-part-2-dqn-improvements-d3563f665a2c
9. https://github.com/keras-rl/keras-rl
