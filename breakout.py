import gym
from DQNAgent import *

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

# 210 -> 80
def downsample(img):
    return img[::2, ::2]

def preprocess(img):
    return to_grayscale(downsample(img))


def main():
    # for image data
    #BreakoutDeterministic-v4
    #env = gym.make("Breakout-v0")
    #env = gym.make('BreakoutDeterministic-v4')

    # for data in ram
    env = gym.make("Breakout-v4")

    # returns a tuple, grab the first aspect (128)
    input_shape = env.observation_space.shape[0]

    # 4 total actions
    action_space = env.action_space.n

    # choose a model!
    model = 'Dense'

    print('RAM input: ', input_shape, 'ACTION output: ', action_space, 'MODEL used: ', model)

    agent = DQNAgent(input_shape, action_space, model)

    # initialize an observation of the game
    frame = env.reset()
    
    # set an environemntal seed
    env.seed(0)

    # flag for whether we die
    done = False


    for epoch in range(0,1000):

        # if we run out of lives or win,
        if done:
            # reset the game
            observation = env.reset()

        # process the frame
        processed_frame = preprocess(frame)

        # pick an action
        action = agent.act(processed_frame)

        # collect the next frame frames, reward, and done flag
        # and act upon the environment by stepping with some action
        next_frame, reward, done, _ = env.step(action)

        # preprocess the next frame
        processed_next_frame = preprocess(next_frame)

        # remember these states by adding it to the deque
        agent.remember(processed_frame, action, reward, processed_next_frame, done)

        print(processed_next_frame)

        # if the memory is bigger than the batch size (32)
        if len(agent.memory) > agent.batch_size:
            # pick some of the frames out of the memory deque
            agent.replay(agent.batch_size)

        # set the frame from before
        frame = next_frame

        env.render()        # renders each frame


# execute main
if __name__ == "__main__":
    main()