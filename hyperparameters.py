
hp = {
		'HEIGHT' : 84,
        'WIDTH' : 84,


        # MEMORY PARAMETERS
        'GAMMA' : 0.95,

        'REPLAY_ITERATIONS' : 100,

        'REPLAY_SAMPLE_SIZE' : 256,

		'MAX_EPISODES' : 20,

		# MAX_EPOCHS: defomed as as the maximum number of epochs
		'MAX_EPOCHS' : 10,

		# on what mod epochs should we update the target network?
		'TARGET_UPDATE' : 10000,

		# hyper parameters for the DQN
        'LEARNING_RATE' : 0.001,
        
        'EPSILON' : 1.0,              # exploration rate, start at 100%
        'EPSILON_MIN' : 0.01,         # minimum exploration rate
        'EPSILON_DECAY' : 0.999      # decay rate for exploration on each frame



}