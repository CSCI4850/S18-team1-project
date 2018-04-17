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

# used for pretty printing
line_sep = '+----------------------------------------------------------------------------------------+'

# expand dimensions to (1, 84, 84, 4) from (84, 84, 4)
# use array[0:1,:,:,:]
# normalize 0-255 -> 0.0-1.0 to reduce exploding gradient
def normalize_frames(current_frame_history):
    return np.expand_dims(np.float64(current_frame_history / 255.), axis=0)

# reduces the rgb channels to black and white (3 axis to 2 axis),
# then resizes the (210, 160) to (height, width) default: 84, 84
# then reduces the np array to 0-255 ints for saving space
# finally expands the axis 0 dimension by one for saving into the conv net
def preprocess(img):
    img = np.uint8(resize(rgb2gray(img), (hp['HEIGHT'], hp['WIDTH']), mode='reflect') * 255)
    return img
    #return img.reshape(1,84,84)

# prints statistics at the end of every episode
def print_stats(total_episodes_elapsed, total_frames_elapsed, epsilon, 
                episodic_reward, total_reward, avg_reward, avg_Q, time_elapsed):
    print('\nepisodes elapsed: {0:3d} | '    
          'frames elapsed: {1:6d} | '      
          'epsilon: {2:1.5f}\n'             
          'total reward: {3:3.0f} | '        
          'reward this episode: {4:3.0f} | ' 
          'avg reward/episode: {5:3.3f}\n'          
          'avg Q: {6:1.5f} | '
          'time elapsed : {7:5.5}\n'.format(total_episodes_elapsed, total_frames_elapsed, 
                                     epsilon, total_reward, episodic_reward ,avg_reward, avg_Q, time_elapsed))
    print(line_sep)

# plots a graph of the game
def plot_initial_graph(env):
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("%s" % (env.spec.id))
    clear_output(wait=True)
    display(plt.gcf())

# initializes the beginning of an episode by loading the first frame
# however big the frame skip size is default: 4
# into a historical frame buffer
def init_discrete_frame_skip(past_frames_size, frame):
            
    for i in range(hp['FRAME_SKIP_SIZE']):
        past_frames_size[:, :, i] = preprocess(frame)

def init_sliding_frame_skip(past_frames_size, frame):     
    for i in range(hp['FRAME_SKIP_SIZE']+1):
        past_frames_size[:, :, i] = preprocess(frame)

    

# returns the max number of lives in the game being played
def find_max_lives(env):
    # don't step anywhere, but grab info
    _, _, _, info = env.step(0)
    return info['ale.lives']    # return max lives

# main loop, runs everything
def run_discrete(model, agent, target_agent, memory, env, stats):

    # set an environemntal seed
    env.seed(0)
    np.rand.seed(0)

    # initialize an observation of the game
    current_frame = env.reset()

    # flag for whether we die or win a round
    done = False
    
    total_frame_reward = 0

    # total frames: total number of frames elapsed
    # a frame is one instance of each tick of the clock of the game
    total_frames_elapsed = 0

    # episodes: da full round of the game,
    # from max lives to 0 lives or from a win of the round
    total_episodes_elapsed = 0

    # total running reward:  all rewards between all episodes
    rewards = deque()
    
    # total running Q:  all Q between the next 100 episodes
    total_max_Q = deque(maxlen=100)
    
    # initialize max lives to the maximum
    max_lives = find_max_lives(env)
    
    # current frame history of size 4
    current_frame_history = np.zeros([84, 84, 4], dtype=np.uint8)
    
    # next set of frame history of size 4
    next_frame_history = np.zeros([84, 84, 4], dtype=np.uint8)

    # initialize a greedy-e default: 1.0
    e = hp['INIT_EXPLORATION']
    
    # amount of exploration to decrease each frame
    e_step = (hp['INIT_EXPLORATION'] - hp['MIN_EXPLORATION']) / hp['EXPLORATION']


    # iterate through a total amount of episodes
    for total_episodes_elapsed in range(hp['MAX_EPISODES']):

        if done:
            current_frame = env.reset()     # reset the game
            lives = max_lives     # reset the number of lives we have
            episodic_reward = 0   # reset the episodic reward
            episodic_frame = 0    # reset the episodic frames
            done = False          # reset the done flag
        
        # do nothing for some amount of the initial game
        # makes new episodes slightly different from each other
        for _ in range(random.randint(1, hp['NO-OP_MAX'])):
            current_frame, _, _, _ = env.step(0) # do nothing
        
        # set up the current and next frame with the first frame of the game
        init_discrete_frame_skip(current_frame_history, current_frame)
        init_discrete_frame_skip(next_frame_history, current_frame)

        # while the episode is not done,
        while not done:
            
            total_frame_reward = 0
            
            # e-greedy scaled linearly over time
            # starts at 1.0 ends at 0.1
            if e > hp['MIN_EXPLORATION'] and total_frames_elapsed < hp['EXPLORATION']:
                e -= e_step
                
            # get Q value
            Q = agent.model.predict(normalize_frames(current_frame_history))
            
            # pick an action
            max_Q, action = agent.act(Q, e)

            # increase the total Q value
            total_max_Q.append(max_Q)

            # determine an action every 4 frames
            for i in range (hp['FRAME_SKIP_SIZE']):

                # renders each frame
                #if hp['RENDER_ENV']:
                #    env.render() 
                
                # increase actual total frames elapsed
                total_frames_elapsed += 1

                # collect the next frame frames, reward, and done flag
                # and act upon the environment by stepping with some action
                # increase action 1 to skip no-op and replace with 
                next_frame, reward, done, info = env.step(action + 1)

                # have a running total
                rewards.append(reward)

                # episodic reward
                total_frame_reward += reward
                
                # fill the next frame history
                next_frame_history[:,:,i] = preprocess(next_frame)
            
            episodic_reward += total_frame_reward
            
            # clip the reward between [-1, 1]
            clipped_reward = np.clip(total_frame_reward, -1, 1)
            
            # capture how many lives we now have after taking another step
            # used in place of done in remmeber because an episode is technically
            # only as long as the agent is alive, speeds up training
            current_lives = info['ale.lives']
            # checks whether we have lost a life
            # used to send that into done rather than waiting until an episode is done
            died = lives > current_lives

            # remember the current and next frame with their actions
            memory.remember_discrete(current_frame_history, action, 
                                     clipped_reward, next_frame_history, died)
            
            # set the next frame history
            current_frame_history = next_frame_history
            # set new lives
            lives = current_lives
            
            # if we have begun training
            if total_frames_elapsed > hp['REPLAY_START']:
                memory.replay_discrete(agent.model, target_agent.model)
    
                # target model weights <- model weights
                if total_frames_elapsed % hp['TARGET_UPDATE'] == 0:
                    target_agent.target_update(agent.model)
                 
            
        # end time of the program
        end_episode_time = time.time()

        # total time in seconds
        time_elapsed = end_episode_time - start_time

        total_reward = np.sum(rewards)
        avg_reward_per_episode = total_reward / (total_episodes_elapsed+1)
      
        # record stats
        episode_stats = [total_episodes_elapsed, total_frames_elapsed, episodic_reward, total_reward, avg_reward_per_episode, np.mean(total_max_Q)]
        stats.append(episode_stats)
        
        # prints our statistics
        print_stats(total_episodes_elapsed, total_frames_elapsed, e, episodic_reward, total_reward, avg_reward_per_episode, np.mean(total_max_Q), time_elapsed)
        
        # when to save the model
        if total_episodes_elapsed+1 % hp['SAVE_MODEL'] == 0:
            agent.save()

# main loop, runs everything
def run_frame_sliding(model, agent, target_agent, memory, env, stats):

    # set an environemntal seed
    env.seed(0)
    np.rand.seed(0)

    # initialize an observation of the game
    current_frame = env.reset()

    # flag for whether we die or win a round
    done = False
    
    total_frame_reward = 0

    # total frames: total number of frames elapsed
    # a frame is one instance of each tick of the clock of the game
    total_frames_elapsed = 0

    # episodes: da full round of the game,
    # from max lives to 0 lives or from a win of the round
    total_episodes_elapsed = 0

    # total running reward:  all rewards between all episodes
    rewards = deque()
    
    # total running Q:  all Q between the next 100 episodes
    total_max_Q = deque(maxlen=100)
    
    # initialize max lives to the maximum
    max_lives = find_max_lives(env)
    
    # sliding frame history
    frame_history = np.zeros([84, 84, 5], dtype=np.uint8)

    # current frame history of size 4
    current_frame_history = np.zeros([84, 84, 4], dtype=np.uint8)
    
    # next set of frame history of size 4
    next_frame_history = np.zeros([84, 84, 4], dtype=np.uint8)

    # initialize a greedy-e default: 1.0
    e = hp['INIT_EXPLORATION']
    
    # amount of exploration to decrease each frame
    e_step = (hp['INIT_EXPLORATION'] - hp['MIN_EXPLORATION']) / hp['EXPLORATION']


    # iterate through a total amount of episodes
    for total_episodes_elapsed in range(hp['MAX_EPISODES']):

        if done:
            current_frame = env.reset()     # reset the game
            lives = max_lives     # reset the number of lives we have
            episodic_reward = 0   # reset the episodic reward
            episodic_frame = 0    # reset the episodic frames
            done = False          # reset the done flag
        
        # do nothing for some amount of the initial game
        # makes new episodes slightly different from each other
        for _ in range(random.randint(1, hp['NO-OP_MAX'])):
            current_frame, _, _, _ = env.step(0) # do nothing
        
        # set up the current and next frame with the first frame of the game
        init_sliding_frame_skip(frame_history, current_frame)

        # while the episode is not done,
        while not done:
            
            total_frame_reward = 0
            
            # e-greedy scaled linearly over time
            # starts at 1.0 ends at 0.1
            if e > hp['MIN_EXPLORATION'] and total_frames_elapsed < hp['EXPLORATION']:
                e -= e_step
                
            # get Q value
            Q = agent.model.predict(normalize_frames(frame_history))
            
            # pick an action
            max_Q, action = agent.act(Q, e)

            # increase the total Q value
            total_max_Q.append(max_Q)

            # determine an action every 4 frames
            for i in range (hp['FRAME_SKIP_SIZE']):

                # renders each frame
                #if hp['RENDER_ENV']:
                #    env.render() 
                
                # increase actual total frames elapsed
                total_frames_elapsed += 1

                # collect the next frame frames, reward, and done flag
                # and act upon the environment by stepping with some action
                # increase action 1 to skip no-op and replace with 
                next_frame, reward, done, info = env.step(action + 1)

                # have a running total
                rewards.append(reward)

                # episodic reward
                total_frame_reward += reward
                
                # fill the next frame history
                next_frame_history[:,:,i] = preprocess(next_frame)
            

            # 1, 2, 3, 4 <- 0, 1, 2, 3
            frame_history[:, :, 1:] = next_frame_history

            episodic_reward += total_frame_reward
            
            # clip the reward between [-1, 1]
            clipped_reward = np.clip(total_frame_reward, -1, 1)
            
            # capture how many lives we now have after taking another step
            # used in place of done in remmeber because an episode is technically
            # only as long as the agent is alive, speeds up training
            current_lives = info['ale.lives']
            # checks whether we have lost a life
            # used to send that into done rather than waiting until an episode is done
            died = lives > current_lives

            # remember the current and next frame with their actions
            memory.remember_frame_sliding(frame_history, action, clipped_reward, died)

            # 0, 1, 2, 3 <- 1, 2, 3, 4
            # advance the frame history buffer
            frame_history[:, :, :4] = frame_history[:, :, 1:]
            
            # set new lives
            lives = current_lives
            
            # if we have begun training
            if total_frames_elapsed > hp['REPLAY_START']:
                memory.replay_slidding(agent.model, target_agent.model)
    
                # target model weights <- model weights
                if total_frames_elapsed % hp['TARGET_UPDATE'] == 0:
                    target_agent.target_update(agent.model)

            
        # end time of the program
        end_episode_time = time.time()

        # total time in seconds
        time_elapsed = end_episode_time - start_time

        total_reward = np.sum(rewards)
        avg_reward_per_episode = total_reward / (total_episodes_elapsed+1)
      
        # record stats
        episode_stats = [total_episodes_elapsed, total_frames_elapsed, episodic_reward, total_reward, avg_reward_per_episode, np.mean(total_max_Q)]
        stats.append(episode_stats)
        
        # prints our statistics
        print_stats(total_episodes_elapsed, total_frames_elapsed, e, episodic_reward, total_reward, avg_reward_per_episode, np.mean(total_max_Q), time_elapsed)
        
        # when to save the model
        if total_episodes_elapsed+1 % hp['SAVE_MODEL'] == 0:
            agent.save()


def main():

    # start time of the program
    start_time = time.time()

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
          'ACTION output:', action_space, 'DISCRETE FRAME SAVING:', hp['DISCRETE_FRAMING'])

    # performance
    stats = []

    # create a DQN Agent
    agent = DQNAgent(new_input_space, action_space, model)
    
    # to load weights
    if (hp['LOAD_WEIGHTS']):
        agent.load(hp['LOAD_WEIGHTS'])

    # and a target DQN Agent
    target_agent = DQNAgent(new_input_space, action_space, model)

    # create a memory for remembering and replay
    memory = ReplayMemory(hp['MEMORY_SIZE'], new_input_space, action_space)

    if hp['DISCRETE_FRAMING']:
        # run the main loop of the game
        run_discrete(model, agent, target_agent, memory, env, stats)

    else:
        run_frame_sliding(model, agent, target_agent, memory, env, stats)

    # end time of the program
    end_time = time.time()

    # total time in seconds
    time_elapsed = end_time - start_time

    # print the final time elapsed
    print('finished training in', time_elapsed, 'seconds')

    # save and quit
    agent.quit(mean_times, stats)


# execute main
if __name__ == "__main__":
    main()