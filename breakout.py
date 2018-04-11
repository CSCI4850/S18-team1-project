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
    return img.reshape(1,84,84)


def print_stats(total_episodes_elapsed, total_frames_elapsed, epsilon, episodic_reward, total_reward, avg_reward, avg_Q):
    print('total episodes elapsed:', total_episodes_elapsed,
          'total frames elapsed:',   total_frames_elapsed,
          'epsilon:',                epsilon,
          'total reward:',           total_reward,
          'reward this episode:',    episodic_reward,
          'avg reward:',             avg_reward,
          'avg Q:',                  avg_Q)

def plot_initial_graph(env):
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("%s" % (env.spec.id))
    clear_output(wait=True)
    display(plt.gcf())


def init_frame_skip(past_frames_size, frame):
    for i in range(hp['FRAME_SKIP_SIZE']):
        past_frames_size[:, :, i] = preprocess(frame)



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
    total_reward = 0

    # total episodic reward: total reward for a certain episode
    episodic_reward = 0

    # total frames elapsed in an episode
    episodic_frame = 0

    current_frame_history = np.zeros([84, 84, 4], dtype=np.uint8)

    next_frame_history = np.zeros([84, 84, 4], dtype=np.uint8)


    total_Q = 0

    # change if we ever use a different model
    if model is 'Convolutional':


        # iterate through a total amount of episodes
        for total_episodes_elapsed in range(hp['MAX_EPISODES']):

            # when to save the model
            if total_episodes_elapsed % hp['SAVE_MODEL'] == 0 and total_episodes_elapsed is not 0:
                agent.save()


            # when we run out of lives or win a round
            if done:

                env.reset()           # reset the game
                episodic_reward = 0   # reset the episodic reward
                episodic_frame = 0    # reset the episodic frames
                total_Q = 0           # reset the runnning total for the Q value
                done = False          # reset the done flag


                current_frame_history = np.zeros([84, 84, 4], dtype=np.uint8)
                next_frame_history = np.zeros([84, 84, 4], dtype=np.uint8)

            init_frame_skip(current_frame_history, current_frame)
            init_frame_skip(next_frame_history, current_frame)


            # while the episode is not done,
            while not done:

                # process the frame
                processed_current_frame = preprocess(current_frame)

                # add the new processed frame to the last slot of the depth
                current_frame_history[:, :, 3] = processed_current_frame

                # determine an action every 4 frames
                if episodic_frame % hp['FRAME_SKIP_SIZE'] == 0:
                    # get Q value
                    Q = agent.model.predict(np.expand_dims(current_frame_history, axis=0))

                    # pick an action
                    Q, next_4_frame_action = agent.act(Q)

                    # increase the total Q value
                    total_Q += Q

                # collect the next frame frames, reward, and done flag
                # and act upon the environment by stepping with some action
                next_frame, reward, done, info = env.step(next_4_frame_action)

                # have a running total
                total_reward += reward

                # episodic reward
                episodic_reward += reward

                # preprocess the next frame
                processed_next_frame = preprocess(next_frame)

                # add the next frame
                next_frame_history[:, :, 3] = processed_next_frame


                # remember the current and next frame with their actions
                memory.remember(current_frame_history, next_4_frame_action, reward, next_frame_history, done)

                # increase the frame counter
                total_frames_elapsed += 1
                episodic_frame += 1

                # set the frame from before
                current_frame = next_frame
                
                current_frame_history[:, :, :3] = current_frame_history[:, :, 1:]
                next_frame_history[:, :, :3] = next_frame_history[:, :, 1:]

                # when to learn and replay to update the model
                if total_frames_elapsed % hp['TARGET_UPDATE'] == 0:
                    # update the target model
                    memory.replay(agent.model, target_agent.model)


                if hp['RENDER_ENV'] is True:
                    env.render()        # renders each frame

            # record
            times_window.append(episodic_frame)
            mean_time = np.mean(times_window)
            mean_times.append(mean_time)

            avg_reward = total_reward/total_frames_elapsed
            avg_Q = total_Q/(total_frames_elapsed / hp['FRAME_SKIP_SIZE'])

            # prints our statistics
            print_stats(total_episodes_elapsed, total_frames_elapsed, hp['EPSILON'], episodic_reward, total_reward, avg_reward, avg_Q)


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
