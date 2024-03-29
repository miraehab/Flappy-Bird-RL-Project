import flappy_bird_gym
import random
import numpy as np
from collections import deque
from keras.layers import Input, Dense
from keras.models import load_model, save_model, Sequential
from keras.optimizers import RMSprop

# Neural Network for the Agent
def NeuralNetwork(input_shape, output_shape):
    model = Sequential()
    #we will not use Conv layer as here we use a ram environment not a pixelated environment
    model.add(Dense(512, input_shape=input_shape, activation='relu',  kernel_initializer='he_uniform'))
    model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    # activation = 'linear' -> so we can squish our computation that we did previously into one of the possible output variables
    # so the output variable will be an action (0-> nothing, 1-> jump)
    model.add(Dense(output_shape, activation='linear', kernel_initializer='he_uniform'))

    model.compile(loss='mse', optimizer=RMSprop(lr=0.0001, rho=0.95, epsilon=0.01), metrics=['accuracy'])
    model.summary()

    return model

# Brain of Agent | BluePrint of Agent
class DQNAgent:
    def __init__(self):

        # ****** Environment Variables ******
        self.env = flappy_bird_gym.make("FlappyBird-v0")
        self.episodes = 1000
        self.state_space = self.env.observation_space.shape[0]
        self.action_scpace = self.env.action_space.n
        #will act as the database that the agent will learn from
        #as the environment gets more comlicated we may want to increase the max_len
        self.memory = deque(maxlen=2000)


        # ****** Hyperparameters ******
        #the discount rate (the priority that we put for immediate reward)
        #0.95 - > we will use heigh priority for immediate reward and don't care too much on future reward
        self.gamma = 0.95
        # epsilon -> is the probabilty of taking a random action
        # 1 -> we will start with using random action and we will decay this probability (epsilon) as we poceed in the training
        self.epsilon = 1
        self.epsilon_decay = 0.9999
        #the minimum probabilty of taking a random action
        self.epsilon_min = 0.01
        #batch_num -> the amount of datapoints that we will input to our neural network for trainning 
        # a heigher number is normally better as it will be able to find patterns in the data 
        self.batch_num = 64 #16, 32, 64, 128, 256


        self.train_start = 1000
        self.jump_prob = 0.01
        self.model = NeuralNetwork(input_shape=(self.state_space,), output_shape=self.action_scpace)

    def act(self, state):
        
        #if we want to predict our action based on our model
        if np.random.random() > self.epsilon:
            return np.argmax(self.model.predict(state))
        return 1 if np.random.random() < self.jump_prob else 0
    
    def learn(self):
        #warm up rounds
        # we don't want to learn yet we still need more data
        if len(self.memory) < self.train_start:
            return
        
        # take a batch of our memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_num))

        #Variables to store mini-batch info
        #create our state and next state
        state = np.zeros((self.batch_num, self.state_space))
        next_state = np.zeros((self.batch_num, self.state_space))

        action, reward, done = [], [], []

        # to create the mini batch that we took from memory
        for i in range(self.batch_num):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state = minibatch[i][3]
            done.append(minibatch[i][4])

        #Using supervised Learning to predict the target (so we have a value that we could compare to)
        target = self.model.predict(state)
        target_next = self.model.predict(state)

        # Q-Learning Part
        #building the true values as we can 
        for i in range(self.batch_num):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))


        # train the model using the predited data
        self.model.fit(state, target, batch_size=self.batch_num, verbose=0)


        

        
    
    def train(self):
        for i in  range(self.episodes):
            state = self.env.reset()
            # we want to reshape our state in manner that the neural network could understand.
            state = np.reshape(state, [1, self.state_space])
            done = False
            score = 0
            #decay the epsilon
            self.epsilon = self.epsilon*self.epsilon_decay if self.epsilon*self.epsilon_decay > self.epsilon_min else self.epsilon_min

            while not done:
                self.env.render()
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)

                #reshape next state as we reshaped the state above
                next_state = np.reshape(next_state, [1, self.state_space])
                # because in this env (flappy bird) the more you go the more you get score 
                score += 1

                # we want to punish the agent every time it dies
                if done:
                    reward -= 100
                
                self.memory.append((state, action, reward, next_state, done))
                state = next_state

                if done:
                    print("Episode: {}\nScore: {}\nEpsilon: {:.2}".format(i, score, self.epsilon))
                    #save model
                    if score > 1000:
                        self.model.save("flappy_agent_1.h5")
                        return 

                # The function that will train the neural network
                self.learn()
                 


if __name__ == '__main__':
    agent = DQNAgent()
    agent.train()

    """
    To Load the model:
    self.model = load_model("flappy_agent_1.h5")
    state = np.reshape(state, [1, self.state_space])
    action = self.model.predict(state)
            ____ step_____
    state = np.reshape(next_state, [1, self.state_space])
    # to visualize the score only not to train
    score += 1

    if done:
    break

    """
