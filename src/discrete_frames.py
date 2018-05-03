"""
    Runs two discrete frame windows
"""

import numpy as np
from collections import deque
import random
import time


from utils import *
from hyperparameters import *


# main loop, runs everything
def run_discrete(agent, target_agent, memory, env, stats, start_time):

    # initialize an observation of the game
    current_frame = env.reset()

    # flag for whether we die or win a round
    done = False
    
    total_episodic_reward = episodic_reward = 0

    # total frames: total number of frames elapsed
    # a frame is one instance of each tick of the clock of the game
    total_frames_elapsed = 0

    # episodes: da full round of the game,
    # from max lives to 0 lives or from a win of the round
    total_episodes_elapsed = 0

    # total running reward:  all rewards between all episodes
    rewards = np.zeros([hp['MAX_EPISODES']], dtype=np.uint16)

    # total running Q:  all Q between the next 100 episodes
    total_max_Q = deque(maxlen=100)
    
    # initialize max lives to the maximum
    max_lives = lives = find_max_lives(env)
    
    # current frame history
    current_frame_history = np.zeros([1, hp['HEIGHT'], hp['WIDTH'], hp['FRAME_BATCH_SIZE']])
    
    # next set of frame history
    next_frame_history = np.zeros([1, hp['HEIGHT'], hp['WIDTH'], hp['FRAME_BATCH_SIZE']])

    # initialize a greedy-e default: 1.0
    e = hp['INIT_EXPLORATION']
    
    # amount of exploration to decrease each frame
    e_step = (hp['INIT_EXPLORATION'] - hp['MIN_EXPLORATION']) / hp['EXPLORATION']

    try:
        # iterate through a total amount of episodes
        for total_episodes_elapsed in range(hp['MAX_EPISODES']):
            
            if total_frames_elapsed > hp['MAX_FRAMES']:
                break
            
            if done:
                current_frame = env.reset()     # reset the game
                lives = max_lives     # reset the number of lives we have
                episodic_reward = 0   # reset the episodic reward
                total_episodic_reward = 0
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
                                
                # get Q value
                Q = agent.model.predict(normalize_frames(current_frame_history[:, :, :, :]))
                
                # pick an action
                max_Q, action = agent.act(Q, e)

                # increase the total Q value
                total_max_Q.append(max_Q)

                # determine an action every 4 frames
                # for i in range (hp['FRAME_SKIP_SIZE']):

                # e-greedy scaled linearly over time
                # starts at 1.0 ends at 0.1
                if e > hp['MIN_EXPLORATION'] and total_frames_elapsed < hp['EXPLORATION']:
                    e -= e_step
                
                # renders each frame
                if hp['RENDER_ENV']:
                    env.render() 

                # collect the next frame frames, reward, and done flag
                # and act upon the environment by stepping with some action
                # increase action 1 to skip no-op and replace with 
                next_frame, reward, done, info = env.step(action + 1)

                # increase actual total frames elapsed
                total_frames_elapsed += 1

                # episodic reward
                total_episodic_reward += reward
                episodic_reward += reward
                
                # fill the next frame history
                next_frame_history[:, :, :, hp['FRAME_BATCH_SIZE']-1] = preprocess(next_frame)

                # capture how many lives we now have after taking another step
                # used in place of done in remmeber because an episode is technically
                # only as long as the agent is alive, speeds up training
                current_lives = info['ale.lives']
                # checks whether we have lost a life
                # used to send that into done rather than waiting until an episode is done
                if lives > current_lives:
                    died = 1
                    episodic_reward = 0
                else:
                    died = 0
                                
                # clip the reward between [-1.0, 1.0]
                clipped_reward = np.clip(episodic_reward, -1.0, 1.0)
                
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
        
            # have a running total
            rewards[total_episodes_elapsed] = total_episodic_reward        
                
            # end time of the program
            end_episode_time = time.time()

            # total time in seconds
            time_elapsed = end_episode_time - start_time

            # calculate the total and average reward
            total_reward = np.sum(rewards)
            last_25_reward = np.sum(rewards[total_episodes_elapsed - 25:total_episodes_elapsed])
            avg_reward_per_episode = last_25_reward / 25
          
            # record stats
            episode_stats = [total_episodes_elapsed, 
                             total_frames_elapsed, 
                             total_episodic_reward, 
                             total_reward, 
                             avg_reward_per_episode, 
                             np.mean(total_max_Q)]

            stats.append(episode_stats)
            
            # prints our statistics
            print_stats(total_episodes_elapsed, 
                        total_frames_elapsed, 
                        e, 
                        total_episodic_reward, 
                        total_reward, 
                        avg_reward_per_episode, 
                        np.mean(total_max_Q), 
                        time_elapsed)
            
            # when to save the model
            if (total_episodes_elapsed+1) % hp['SAVE_MODEL'] == 0:
                agent.save_weights()
                agent.save_stats(stats)

    except KeyboardInterrupt:   
        agent.save_weights()
        agent.save_stats(stats)
