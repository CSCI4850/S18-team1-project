###-----------------------------------------###
###          Breakout Main Loop             ###
###-----------------------------------------###

import gym

import time

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

def plot_initial_graph(env):
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("%s" % (env.spec.id))
    clear_output(wait=True)
    display(plt.gcf())


def run(model, agent, target_agent, memory, env):
    
    # initialize an observation of the game
    frame = state = env.reset()
    
    # set an environemntal seed
    env.seed(0)

    # flag for whether we die or win a round
    done = False

    # reward: reward for a particular frame
    reward = 0

    # total frames: total number of frames elapsed
    # a frame is one instance of each tick of the clock of the game
    total_frames_elapsed = 0

    # episodes: da full round of the game,
    # from max lives to 0 lives or from a win of the round
    total_episodes_elapsed = 0

    # total running reward:  all rewards between all episodes
    total_reward = 0

    # total episodic reward: total reward for a certain episode
    episodic_reward = 0


    if model is 'Convolutional':


        # iterate through a total amount of episodes
        while total_episodes_elapsed < hp['MAX_EPISODES']:


            # when to save the model
            if total_episodes_elapsed % hp['SAVE_MODEL'] == 0 and total_episodes_elapsed is not 0:
                agent.save()

            if total_frames_elapsed % hp['TARGET_UPDATE'] == 0 and total_frames_elapsed is not 0:
                # update the target model
                pass

            # increase the number of episodes counter
            total_episodes_elapsed += 1

            # when we run out of lives or win a round
            if done: 
                
                env.reset()           # reset the game
                episodic_reward = 0   # reset the episodic reward
                done = False          # reset the done flag

            # while the episode is not done,
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
                memory.replay(agent, target_agent)

                # apply exploration decay if epsilon is greater than epsilon min
                if hp['EPSILON'] > hp['EPSILON_MIN']:
                    hp['EPSILON'] *= hp['EPSILON_DECAY']

                # set the frame from before
                frame = next_frame

                if hp['RENDER_ENV'] is True:
                    env.render()        # renders each frame

            # prints our statistics
            print_stats(total_episodes_elapsed, total_frames_elapsed, episodic_reward, total_reward)


def main():

    # start time of the program
    start_time = time.time()


    model = 'Convolutional'

    # pixel/frame data
    env = gym.make("Breakout-v4")


    # 4 actions
    # 0: no-op 1: fire 2: right 3: left
    action_space = env.action_space.n

    # returns a tuple, (210, 160, 3)
    input_space = env.observation_space.shape[0]

    # create a new 3 dimensional space for a downscaled grayscale image
    new_input_space = np.array([hp['HEIGHT'], hp['WIDTH'], 1])


    # print the initial state
    print('FRAME input:', input_space, 'NEW input:', new_input_space,  
          'ACTION output:', action_space, 'MODEL used:', model)


    # create a DQN Agent
    agent = DQNAgent(new_input_space, action_space, model)

    # and a target DQN Agent
    target_agent = DQNAgent(new_input_space, action_space, model)

    # create a memory for remembering and replay
    memory = ReplayMemory(hp['MEMORY_SIZE'], new_input_space, action_space)
   
    # run the main loop of the game
    run(model, agent, target_agent, memory, env)

    # end time of the program
    end_time = time.time()

    # total time in seconds
    time_elapsed = end_time - start_time

    # print the final time elapsed
    print('finished training in', time_elapsed, 'in seconds')


# execute main
if __name__ == "__main__":
    main()