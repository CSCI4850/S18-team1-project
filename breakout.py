###-----------------------------------------###
###          Breakout Main Loop             ###
###-----------------------------------------###

import gym

import time

from DQNAgent import *
from ReplayMemory import *
from hyperparameters import *


from collections import deque

from skimage.transform import resize
from skimage.color import rgb2gray
from keras.preprocessing import image

line_sep = '+----------------------------------------------------------------------------------------+'

def normalize_frames(current_frame_history):
    # expand dimensions to (1, 84, 84, 5) from (84, 84, 5)
    # normalize 0-255 -> 0-1 to reduce exploding gradient
    return np.expand_dims(np.float64(current_frame_history / 255.), axis=0)

def preprocess(img):
    img = np.uint8(resize(rgb2gray(img), (hp['HEIGHT'], hp['WIDTH']), mode='reflect') * 255)
    return img.reshape(1,84,84)


def print_stats(total_episodes_elapsed, total_frames_elapsed, epsilon, 
                episodic_reward, total_reward, avg_reward, avg_Q,
                episodic_avg_reward):
    print('\nepisodes elapsed: {0:3d} | '    
          'frames elapsed: {1:6d} | '      
          'epsilon: {2:1.5f}\n'             
          'total reward: {3:3.0f} | '        
          'episodic reward: {4:3.0f} | ' 
          'episodic avg reward: {7:3.2f} | '
          'total avg reward: {5:3.3f}\n'             
          'avg Q: {6:1.5f}\n'.format(total_episodes_elapsed, total_frames_elapsed, 
                            epsilon, total_reward, episodic_reward ,avg_reward, avg_Q,
                            episodic_avg_reward))
    print(line_sep)

def plot_initial_graph(env):
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("%s" % (env.spec.id))
    clear_output(wait=True)
    display(plt.gcf())


def init_frame_skip(past_frames_size, frame):
    for i in range(hp['FRAME_SKIP_SIZE']):
        past_frames_size[:, :, i] = preprocess(frame)
        
def find_max_lives(env):
    # don't step anywhere, but grab info
    _, _, _, info = env.step(0)
    return info['ale.lives']    # return max lives


def check_lives(life, current_life):
    return ( True if life > current_life else False )



def run(model, agent, target_agent, memory, env, mean_times):

    # Peformance stats
    times_window = deque(maxlen=100)

    # initialize an observation of the game
    current_frame = env.reset()

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
    total_reward = deque()
    
    # total running Q:  all Q between all episodes
    total_Q = deque()

    # total episodic reward: total reward for a certain episode
    episodic_reward = 0

    # total frames elapsed in an episode
    episodic_frame = 0
    
    e = hp['INIT_EXPLORATION']
    
    # initialize lives to the maximum
    lives = max_lives = find_max_lives(env)

    frame_history = np.zeros([84, 84, 5], dtype=np.uint8)

    current_frame_history = np.zeros([84, 84, 4], dtype=np.uint8)

    next_frame_history = np.zeros([84, 84, 4], dtype=np.uint8)

    # change if we ever use a different model
    if model is 'Convolutional':


        # iterate through a total amount of episodes
        for total_episodes_elapsed in range(hp['MAX_EPISODES']):

            # when we run out of lives or win a round
            if done:

                env.reset()           # reset the game
                lives = max_lives     # reset the number of lives we have
                episodic_reward = 0   # reset the episodic reward
                episodic_frame = 0    # reset the episodic frames
                done = False          # reset the done flag


            init_frame_skip(current_frame_history, current_frame)
            init_frame_skip(next_frame_history, current_frame)


            # while the episode is not done,
            while not done:
                
                # e-greedy
                if e > hp['MIN_EXPLORATION'] and total_frames_elapsed > hp['REPLAY_START']:
                    e -= (hp['INIT_EXPLORATION'] - hp['MIN_EXPLORATION']) / hp['EXPLORATION']

                # determine an action every 4 frames
                if episodic_frame % hp['FRAME_SKIP_SIZE'] == 0:
                    # get Q value
                    Q = agent.model.predict(normalize_frames(current_frame_history))

                    # pick an action
                    Q, next_4_frame_action = agent.act(Q, e)

                    # increase the total Q value
                    total_Q.append(Q)


                # collect the next frame frames, reward, and done flag
                # and act upon the environment by stepping with some action
                next_frame, reward, done, info = env.step(next_4_frame_action)

                # have a running total
                total_reward.append(reward)

                # episodic reward
                episodic_reward += reward
                
                # clip the reward between [-1, 1]
                # may or may not affect breakout
                reward = np.clip(reward, -1, 1)
                
                # capture how many lives we now have after taking another step
                current_lives = check_lives(lives, info['ale.lives'])

                # preprocess the next frame
                processed_next_frame = preprocess(next_frame)

                # add the next frame, 4th item is new frame
                frame_history[:, :, 4] = processed_next_frame

                # [1, 2, 3, 4], 4 frames
                next_frame_history = frame_history[:, :, 1:]

                # remember the current and next frame with their actions
                memory.remember(current_frame_history, next_4_frame_action, reward, next_frame_history, current_lives)

                # increase the frame counter
                total_frames_elapsed += 1
                episodic_frame += 1

                # current is now the next frame
                current_frame_history = next_frame_history
                # update frame history
                # [0, 1, 2, 3] <- [1, 2, 3, 4]
                frame_history[:, :, :4] = frame_history[:, :, 1:]
    
                if total_frames_elapsed > hp['REPLAY_START']:
                    memory.replay(agent.model, target_agent.model)
        
                    # when to learn and replay to update the model
                    if total_frames_elapsed % hp['TARGET_UPDATE'] == 0:
                        # update the target model
                        print('HELLO?')
                        target_agent.target_udpate(agent.model)

                if hp['RENDER_ENV'] is True:
                    env.render()        # renders each frame

            # record
            times_window.append(episodic_frame)
            mean_time = np.mean(times_window)
            mean_times.append(mean_time)

            # prints our statistics
            print_stats(total_episodes_elapsed, total_frames_elapsed, e, episodic_reward, np.sum(total_reward), np.mean(total_reward), np.mean(total_Q), episodic_reward/(total_episodes_elapsed+1))
            
            # when to save the model
            if total_episodes_elapsed % hp['SAVE_MODEL'] == 0 and total_episodes_elapsed is not 0:
                agent.save()


def main():

    # start time of the program
    start_time = time.time()

    model = 'Convolutional'

    # pixel/frame data
    env = gym.make(hp['GAME'])

    # 4 actions
    # 0: no-op 1: fire 2: right 3: left
    action_space = env.action_space.n

    # returns a tuple, (210, 160, 3)
    input_space = env.observation_space.shape[0]

    # create a new 3 dimensional space for a downscaled grayscale image
    new_input_space = np.array([hp['HEIGHT'], hp['WIDTH'], hp['FRAME_SKIP_SIZE']], dtype=np.uint8)

    # print the initial state
    print('FRAME input:', input_space, 'NEW input:', new_input_space,
          'ACTION output:', action_space, 'MODEL used:', model)

    # performance
    mean_times = deque(maxlen=hp['MAX_EPISODES'])

    # create a DQN Agent
    agent = DQNAgent(new_input_space, action_space, model)

    if (hp['LOAD_WEIGHTS']):
        agent.load(hp['LOAD_WEIGHTS'])

    # and a target DQN Agent
    target_agent = DQNAgent(new_input_space, action_space, model)

    # create a memory for remembering and replay
    memory = ReplayMemory(hp['MEMORY_SIZE'], new_input_space, action_space)

    # run the main loop of the game
    run(model, agent, target_agent, memory, env, mean_times)

    # end time of the program
    end_time = time.time()

    # total time in seconds
    time_elapsed = end_time - start_time

    # print the final time elapsed
    print('finished training in', time_elapsed, 'in seconds')

    # save and quit
    agent.quit(mean_times)


# execute main
if __name__ == "__main__":
    main()
