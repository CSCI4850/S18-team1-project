###--------------------------------------------###
### Hyper Parameter dictionary for O(1) lookup ###
###--------------------------------------------###

hp = {

        ###-----------------------------------------###
        ### hyper parameters for the Main Game Loop ###
        ###-----------------------------------------###
    
        'GAME' : 'BreakoutDeterministic-v4',   # Name of which game to use
                                             # v1-4 Deterministic or NoFrameskip 

        'DISCRETE_FRAMING' : False,      # 2 discrete sets of frames stored in memory
    
        'LOAD_WEIGHTS' : '',           # Loads weights into the model if so desired
                                       # leave '' if starting from a new model

        'RENDER_ENV' : True,          # shows the screen of the game as it learns
                                       # massivly slows the training down when True
                                       # default: False

        'HEIGHT' : 84,                 # height in pixels
        'WIDTH'  : 84,                 # and width in pixels that the game window will get downscaled to
                                       # defaults: 84, 84

        'FRAME_BATCH_SIZE' : 4,        # how many frames we skip and and how many times we choose an action for
                                       # that many frames. (taken out for now)
                                       # default: 4
    
        'MAX_EPISODES' : 50000,        # defined as how many cycles of full life to end life or
                                       # winning a round
                                       # default: 12,000

        'MAX_FRAMES' : 50000000,       # max number of frames allowed to pass before stopping
                                       # default: 50,000,000

        'SAVE_MODEL' : 500,            # how many episodes should we go through until we save the model?
                                       # default: whenever

        'TARGET_UPDATE' : 10000,       # on what mod epochs should we update the target network?
                                       # default: 10000

        ###-----------------------------------###
        ### hyper parameters for the DQNAgent ###
        ###-----------------------------------###
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
        
        'NO-OP_MAX' : 30,              # how many times no-op can be called in a single episode
                                       # chooses a different action if exceeded
                                       # default: 30
                                    

        ###----------------------------------------###
        ### hyper parameters for the Replay Memory ###
        ###----------------------------------------###
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

}
