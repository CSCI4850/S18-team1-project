###--------------------------------------------###
### Hyper Parameter dictionary for O(1) lookup ###
###--------------------------------------------###

hp = {

        ###-----------------------------------------###
        ### hyper parameters for the Main Game Loop ###
        ###-----------------------------------------###
    
        'GAME' : 'BreakoutDeterministic-v4',
    
        'LOAD_WEIGHTS' : '',          # Loads weights into the model if so desired
                                      # leave '' if starting from a new model

        'RENDER_ENV' : False,         # shows the screen of the game as it learns
                                      # massivly slows the training down when True
                                      # default: False

        'HEIGHT' : 84,                # height in pixels
        'WIDTH'  : 84,                # and width in pixels that the game window will get downscaled to
                                      # defaults: 84, 84

        'FRAME_SKIP_SIZE' : 4,        
	
        'MAX_EPISODES' : 200,          # defined as how many cycles of full life to end life or
                                      # winning a round
                                      # default: 

        'SAVE_MODEL' : 201,            # how many episodes should we go through until we save the model?
                                      # default:

	'TARGET_UPDATE' : 10000,      # on what mod epochs should we update the target network?
                                      # default: 10000

        ###-----------------------------------###
	### hyper parameters for the DQNAgent ###
        ###-----------------------------------###
        'WATCH_Q' : False,            # watch the Q function and see what decision it picks
                                      # cool to watch

        'LEARNING_RATE' : 0.00025,      # learning rate of the Adam optimizer
                                      # default: 0.00025
        
        'INIT_EXPLORATION' : 1.0,              # exploration rate, start at 100%
        'EXPLORATION' : 1000000,    # decay rate for exploration on each frame
        'MIN_EXPLORATION' : 0.1,
                                      # defaults: 1.0, 0.999, 0.01
    
        'OPTIMIZER' : 'Adam',          # optimizer used
        'MIN_SQUARED_GRADIENT' : 0.01, # minimum exploration rate
        'MOMENTUM' : 0.95,

        'LOSS' : 'logcosh',           # can be 'logcosh' for logarithm of hyperbolic cosine
                                      # or 'mse' for mean squared error
                                      # default: logcosh

        ###----------------------------------------###
        ### hyper parameters for the Replay Memory ###
        ###----------------------------------------###
        'SHOW_FIT' : 0,                # shows the fit of the model and it's work, turn to 0 for off
                                       # default: 0 for off
    
        'REPLAY_START' : 1000,        # when to start using replay to update the model
                                       # default: 50000 frames

        'MEMORY_SIZE' : 1000000,        # size of the memory bank
                                       # default: 1,000,000

        'GAMMA' : 0.99,                # integration of rewards, discount factor, 
                                       # preference for present rewards as opposed to future rewards
                                       # default: 0.99
        # 4 * 8 = 32 batch
        'REPLAY_ITERATIONS' : 4,     # how mant irerations of replay
                                       # default: 4

        'REPLAY_SAMPLE_SIZE' : 8     # how many frames from the state memory should we use
                                       # default: 8

}