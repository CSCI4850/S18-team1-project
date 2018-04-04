import gym
from DQNAgent import *

from skimage.transform import resize
from skimage.color import rgb2gray
from keras.preprocessing import image

HEIGHT = 84
WIDTH = 84

def preprocess(img):
    return np.uint8(resize(rgb2gray(img), (HEIGHT, WIDTH), mode='reflect') * 255)

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

def print_stats(epoch, done, action, reward, TOTAL_REWARD):
    print('epoch:', epoch, 'done:', done,
          'action:', find_action(action), 'reward:', reward, 
          'total reward:', TOTAL_REWARD)

def run(model, agent, env):
    # initialize an observation of the game
    frame = state = env.reset()
    
    # set an environemntal seed
    env.seed(0)

    # flag for whether we die
    done = False

    TOTAL_REWARD = 0

    MAX_FRAMES = 100000000

    reward = 0

    # episodes: defined as a full round of the game,
    # from max lives to 0 lives or from a win of the round
    episodes = 0

    # frames: defined as the total number of frames elapsed
    # a frame is one instance of each tick of the clock of the game
    frames = 0

    # epochs: defined as the total number of epochs we train through
    # starts at 0, ends at MAX_EPOCHS
    epochs = 0

    # MAX_EPOCHS: defomed as as the maximum number of epochs
    MAX_EPOCHS = 10

    # max lives: starts at 5
    #max_life = find_max_lifes(env)


    if model is 'Convolutional':

        while epochs < MAX_EPOCHS:

            # increase the number of episodes counter
            episodes += 1

            # when we run out of lives or win a round
            if done: 
                # reset the game
                observation = env.reset()


            while not done:

                # increase the frame counter
                frames += 1

                # process the frame
                processed_frame = preprocess(frame)

                # pick an action
                action = agent.act(processed_frame)


                # collect the next frame frames, reward, and done flag
                # and act upon the environment by stepping with some action
                next_frame, reward, done, info = env.step(action)

                # have a running total
                TOTAL_REWARD += reward

                # preprocess the next frame
                processed_next_frame = preprocess(next_frame)

                # remember these states by adding it to the deque
                agent.remember(processed_frame, action, reward, processed_next_frame, done)


                # if the memory is bigger than the batch size (32)
                if len(agent.memory) > agent.batch_size:
                    # pick some of the frames out of the memory deque
                    agent.replay(agent.batch_size)

                # set the frame from before
                frame = next_frame
            

            # prints our statistics
            print_stats(epoch, done, action, reward, TOTAL_REWARD)


            if epochs % 2 == 0 and epochs is not 0:
                model.save()
                #env.render()        # renders each frame

def main():

    model = 'Convolutional'

    if model is 'Convolutional':

        # pixel/frame data
        env = gym.make("Breakout-v4")

        # returns a tuple, (210, 160, 3)
        input_shape = env.observation_space.shape


    # 4 total actions
    action_space = env.action_space.n

    print('FRAME input:', input_shape, 'ACTION output:', action_space, 'MODEL used:', model)

    agent = DQNAgent(input_shape, action_space, model)
   

    # run the main loop of the game
    run(model, agent, env)


# execute main
if __name__ == "__main__":
    main()