import gym




def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

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
    env = gym.make("Breakout-ram-v4")
    observation = env.reset() # This gets us the image
    
    # set an environemntal seed
    env.seed(0)

    # flag for whether we die
    done = False

    # total frames elapsed from the start
    frames_elapsed = 0

    while frames_elapsed in range(0,100):

        # if we lose, reset the environment
        if done:
            env.reset()

        # collect the frames, reward, and done flag
        frame, reward, done, _ = env.step(env.action_space.sample())



        #print(frame)

        env.render()        # renders each frame


# execute main
if __name__ == "__main__":
    main()