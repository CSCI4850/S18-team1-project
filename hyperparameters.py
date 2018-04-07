###--------------------------------------------###
### Hyper Parameter dictionary for O(1) lookup ###
###--------------------------------------------###

hp = {

        ###-----------------------------------------###
        ### hyper parameters for the Main Game Loop ###
        ###-----------------------------------------###
        'LOAD_WEIGHTS' : '',          # Loads weights into the model if so desired
                                      # leave '' if starting from a new model

        'RENDER_ENV' : False,         # shows the screen of the game as it learns
                                      # massivly slows the training down when True
                                      # default: False

        'HEIGHT' : 84,                # height in pixels
        'WIDTH'  : 84,                # and width in pixels that the game window will get downscaled to
                                      # defaults: 84, 84
	
        'MAX_EPISODES' : 10,          # defined as how many cycles of full life to end life or
                                      # winning a round
                                      # default: 

        'SAVE_MODEL' : 10,            # how many episodes should we go through until we save the model?
                                      # default:

	'TARGET_UPDATE' : 10000,      # on what mod epochs should we update the target network?
                                      # default: 10000

        ###-----------------------------------###
	### hyper parameters for the DQNAgent ###
        ###-----------------------------------###
        'WATCH_Q' : False,            # watch the Q function and see what decision it picks
                                      # cool to watch

        'LEARNING_RATE' : 0.001,      # learning rate of the Adam optimizer
                                      # default: 0.001
        
        'EPSILON' : 1.0,              # exploration rate, start at 100%
        'EPSILON_DECAY' : 0.999,      # decay rate for exploration on each frame
        'EPSILON_MIN' : 0.01,         # minimum exploration rate
                                      # defaults: 1.0, 0.999, 0.01

        'LOSS' : 'logcosh',           # can be 'logcosh' for logarithm of hyperbolic cosine
                                      # or 'mse' for mean squared error
                                      # default: logcosh

        ###----------------------------------------###
        ### hyper parameters for the Replay Memory ###
        ###----------------------------------------###
        'SHOW_FIT' : 0,                # shows the fit of the model and it's work, turn to 0 for off
                                       # default: 0 for off

        'MEMORY_SIZE' : 200000,       # size of the memory bank
                                       # default: 1,000,000

        'GAMMA' : 0.95,                # integration of rewards, discount factor, 
                                       # preference for present rewards as opposed to future rewards
                                       # default: 0.95

        'REPLAY_ITERATIONS' : 100,     # how mant irerations of replay
                                       # default: 100

        'REPLAY_SAMPLE_SIZE' : 256     # how many frames from the state memory should we use
                                       # default: 256

}