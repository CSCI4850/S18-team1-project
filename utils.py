"""
   Utility functions used by multiple files

"""

from PIL import Image
from hyperparameters import *
import numpy as np


# used for pretty printing
line_sep = '+----------------------------------------------------------------------------------------+'

# expand dimensions to (1, 84, 84, 4) from (84, 84, 4)
# use array[0:1,:,:,:]
# normalize 0-255 -> 0.0-1.0 to reduce exploding gradient
def normalize_frames(current_frame_history):
    return current_frame_history.astype('float32') / 255.

# reduces the rgb channels to black and white (3 axis to 2 axis),
# then resizes the (210, 160) to (height, width) default: 84, 84
# then reduces the np array to 0-255 ints for saving space
# finally expands the axis 0 dimension by one for saving into the conv net
def preprocess(img):
    # resize and grayscape
    assert img.ndim == 3  # (height, width, channel)
    img = Image.fromarray(img)
    img = img.resize((hp['HEIGHT'], hp['WIDTH'])).convert('L')
    processed_img = np.array(img)
    assert processed_img.shape == (hp['HEIGHT'], hp['WIDTH'])
    return processed_img.astype('uint8')

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
            
    for i in range(hp['FRAME_BATCH_SIZE']):
        past_frames_size[:, :, :, i] = preprocess(frame)

def init_sliding_frame_skip(past_frames_size, frame):     
    for i in range(hp['FRAME_BATCH_SIZE']+1):
        past_frames_size[:, :, :, i] = preprocess(frame)

    
# returns the max number of lives in the game being played
def find_max_lives(env):
    # don't step anywhere, but grab info
    _, _, _, info = env.step(0)
    return info['ale.lives']    # return max lives
