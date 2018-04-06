###-----------------------------------------###
###          Breakout Main Loop             ###
###-----------------------------------------###

import gym
from DQNAgent import *
from ReplayMemory import *
from hyperparameters import *

from skimage.transform import resize
from skimage.color import rgb2gray
from keras.preprocessing import image


def preprocess(img):
    img = np.uint8(resize(rgb2gray(img), (hp['HEIGHT'], hp['WIDTH']), mode='reflect') * 255)
    return img.reshape(1,84,84,1)

def find_action(action):
    # actions:
    # 0: no-op 1: fire 2: right 3: left
    if action is 0:
        return 'no-op'
    elif action is 1:
        return 'fire'
    elif action is 2:
        return 'move right'
    elif action is 3:
        return 'move left'

def print_stats(total_episodes_elapsed, total_frames_elapsed, episodic_reward, total_reward):
    print('total episodes elapsed:', total_episodes_elapsed,
          'total frames elapsed:',   total_frames_elapsed,

          'reward this episode:',    episodic_reward, 
          'total reward:',           total_reward)


def run(model, agent, target_agent, memory, env):
    
    # initialize an observation of the game
    frame = state = env.reset()
    
    # set an environemntal seed
    env.seed(0)

    # flag for whether we die or win a round
    done = False

    # reward:
    reward = 0

    # epochs: defined as the total number of epochs we train through
    # starts at 0, ends at MAX_EPOCHS
    epochs = 0

    # total frames: defined as the total number of frames elapsed
    # a frame is one instance of each tick of the clock of the game
    total_frames_elapsed = 0

    # episodes: defined as a full round of the game,
    # from max lives to 0 lives or from a win of the round
    total_episodes_elapsed = 0

    total_reward = 0

    episodic_reward = 0


    if model is 'Convolutional':

        while total_episodes_elapsed < hp['MAX_EPISODES']:


            if total_episodes_elapsed % 10 == 0 and total_episodes_elapsed is not 0:
                agent.model.save()

            if total_frames_elapsed % hp['TARGET_UPDATE'] == 0 and total_frames_elapsed is not 0:
                # update the target model
                pass

            # increase the number of episodes counter
            total_episodes_elapsed += 1

            # when we run out of lives or win a round
            if done: 
                # reset the game
                env.reset()
                episodic_reward = 0
                done = False


            while not done:

                # increase the frame counter
                total_frames_elapsed += 1


                # process the frame
                processed_frame = preprocess(frame)

                # get Q value
                Q = agent.model.predict(processed_frame)

                # pick an action
                action = agent.act(Q)


                # collect the next frame frames, reward, and done flag
                # and act upon the environment by stepping with some action
                next_frame, reward, done, info = env.step(action)

                # have a running total
                total_reward += reward

                # episodic reward
                episodic_reward += reward

                # preprocess the next frame
                processed_next_frame = preprocess(next_frame)

                # remember these states by adding it to the deque
                memory.replay(agent, target_agent, hp['REPLAY_ITERATIONS'], hp['REPLAY_SAMPLE_SIZE'], hp['GAMMA'])

                if hp['EPSILON'] > hp['EPSILON_MIN']:
                    hp['EPSILON'] *= hp['EPSILON_DECAY']

                # set the frame from before
                frame = next_frame

                env.render()        # renders each frame

            

            # prints our statistics
            print_stats(total_episodes_elapsed, total_frames_elapsed, episodic_reward, total_reward)


def main():

    model = 'Convolutional'

    # pixel/frame data
    env = gym.make("Breakout-v4")


    # 4 actions
    # 0: no-op 1: fire 2: right 3: left
    action_space = env.action_space.n

    # returns a tuple, (210, 160, 3)
    input_space = env.observation_space.shape[0]


    new_input_space = np.array([hp['HEIGHT'], hp['WIDTH'], 1])


    print('FRAME input:', input_space, 'NEW input:', new_input_space,  
          'ACTION output:', action_space, 'MODEL used:', model)


    agent = DQNAgent(new_input_space, action_space, model)
    target_agent = DQNAgent(new_input_space, action_space, model)
    memory = ReplayMemory(hp['MEMORY_SIZE'], new_input_space, action_space)
   
    # run the main loop of the game
    run(model, agent, target_agent, memory, env)


# execute main
if __name__ == "__main__":
    main()