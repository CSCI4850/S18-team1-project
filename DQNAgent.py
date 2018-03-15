from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model

# DQNAgent for breakout
class DQNAgent():
    def __init__(self, input_shape, action_space):

        # input layer into our first dense layer
        self.input_shape = input_shape

        # output layer mapped to an action
        self.action_space = action_space

        # hyper parameters for the DQN
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.batch_size = 32

        self.memory_size = 1000000

        # create replay memory using deque
        self.memory = deque(maxlen=self.memory_size)

        # build the model 
        self.model = self._build_model()

        #self.model.load_weights('pong_weights-18-01-28-16-18.h5')

    def _build_model(self):

        model = Sequential()
        model.add(Dense(24, input_dim=self.input_shape, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
        if not done:
            target = (reward + self.gamma *
            np.amax(self.model.predict(next_state)[0]))
        print(state)
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    ## hard exits the game
    def quit(self):
        # save the model
        fn = 'weights/breakout-v4-ram-weights-' + str(datetime.datetime.now().strftime("%y-%m-%d-%H-%M")) \
             + '.h5'
        self.agent.save(fn)
        print('Saved breakout weights as',fn)
        print('Exiting..')
        pg.quit()
        sys.exit()

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

