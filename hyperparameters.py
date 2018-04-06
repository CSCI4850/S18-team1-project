###--------------------------------------------###
### Hyper Parameter dictionary for O(1) lookup ###
###--------------------------------------------###

hp = {

        ###-----------------------------------------###
        ### hyper parameters for the Main Game Loop ###
        ###-----------------------------------------###
        'HEIGHT' : 84,                # height in pixels
        'WIDTH'  : 84,                # and width in pixels that the game window will get downscaled to

	
        'MAX_EPISODES' : 20,          # defined as how many cycles of full life to end life or
                                      # winning a round

	'TARGET_UPDATE' : 10000,      # on what mod epochs should we update the target network?


        ###-----------------------------------###
	### hyper parameters for the DQNAgent ###
        ###-----------------------------------###
        'LEARNING_RATE' : 0.001,      # learning rate
        
        'EPSILON' : 1.0,              # exploration rate, start at 100%
        'EPSILON_DECAY' : 0.999,      # decay rate for exploration on each frame
        'EPSILON_MIN' : 0.01,         # minimum exploration rate

        'LOSS_FUNCTION' : 'logcosh',   # leave blank if Mean Squared Error is used

        ###----------------------------------------###
        ### hyper parameters for the Replay Memory ###
        ###----------------------------------------###
        'MEMORY_SIZE' : 1000000,        #

        'GAMMA' : 0.95,                 # 

        'REPLAY_ITERATIONS' : 100,      # how mant irerations

        'REPLAY_SAMPLE_SIZE' : 256,     # how many frames from the state memory should we use


}