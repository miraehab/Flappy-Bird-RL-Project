import flappy_bird_gym
import random
import numpy as np
from collections import deque
from keras.layers import Input, Dense
from keras.models import load_model, save_model, Sequential
from keras.optimizers import RMSprop

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
        #self.model = NeuralNet




