import gym
from DQNAgent import *

from skimage.transform import resize
from skimage.color import rgb2gray
from keras.preprocessing import image

HEIGHT = 84
WIDTH = 84

def preprocess(img):
    return np.uint8(resize(rgb2gray(img), (HEIGHT, WIDTH), mode='reflect') * 255)


def pp(observation):
    observation = observation[32:-17, 8:-8]
    img = image.array_to_img(observation, 'channels_last')
    img = img.convert('L')
    img = img.resize((HEIGHT,WIDTH))
    observation = image.img_to_array(img, 'channels_last')
    return np.squeeze(observation)

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


    if model is 'Dense':
        for epoch in range(0,MAX_FRAMES):

            # if we run out of lives or win,
            if done:
                # reset the game
                observation = env.reset()

            # pick an action
            action = agent.act(frame)

            # collect the next frame frames, reward, and done flag
            # and act upon the environment by stepping with some action
            next_state, reward, done, _ = env.step(action)


            # remember these states by adding it to the deque
            agent.remember(state, action, reward, next_state, done)


            # if the memory is bigger than the batch size (32)
            if len(agent.memory) > agent.batch_size:
                # pick some of the frames out of the memory deque
                agent.replay(agent.batch_size)

            # set the state from before
            state = next_state

            #print(processed_next_frame)
            print('action:', find_action(action), 'reward:', reward)

            env.render()        # renders each frame


    if model is 'Convolutional':
        for epoch in range(0,MAX_FRAMES):

            # if we run out of lives or win,
            if done:
                # reset the game
                observation = env.reset()

            # process the frame
            #processed_frame = frame
            processed_frame = preprocess(frame)
            #processed_frame = pp(frame)

            # pick an action
            action = agent.act(processed_frame)


            print('epoch:', epoch, 'done:', done,
                  'action:', find_action(action), 'reward:', reward, 
                  'total reward:', TOTAL_REWARD)

            # collect the next frame frames, reward, and done flag
            # and act upon the environment by stepping with some action
            next_frame, reward, done, info = env.step(action)

            # have a running total
            TOTAL_REWARD += reward

            # preprocess the next frame
            #processed_next_frame = next_frame
            processed_next_frame = preprocess(next_frame)
            #processed_next_frame = pp(next_frame)

            # remember these states by adding it to the deque
            agent.remember(processed_frame, action, reward, processed_next_frame, done)


            # if the memory is bigger than the batch size (32)
            if len(agent.memory) > agent.batch_size:
                # pick some of the frames out of the memory deque
                agent.replay(agent.batch_size)

            # set the frame from before
            frame = next_frame

            #print(processed_next_frame)


            #env.render()        # renders each frame

def main():
    # for image data
    #env = gym.make('BreakoutDeterministic-v4') 


    # choose a model!
    # RAM?
    #model = 'Dense'
    # pixel/frame data?
    model = 'Convolutional'

    if model is 'Dense':

        # ram data
        env = gym.make("Breakout-ram-v4")

        # returns a tuple, 128 bytes
        input_shape = env.observation_space.shape

    elif model is 'Convolutional':

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