import gym
import numpy as np
import random
import matplotlib.pyplot as plt

# Load environment
env = gym.make('FrozenLake-v0')

# Define the neural network mapping 16x1 one hot vector to a vector of 4 Q values
# and training loss
# Neural Network Model
import torch
import torch.nn as nn
from torch.autograd import Variable

input_size = 16
num_classes = 4
learning_rate = 0.01

# Neural Network Model
net = nn.Sequential(nn.Linear(input_size, num_classes, bias=False))


# Loss and Optimizer
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
losses = []
# Implement Q-Network learning algorithm

# Set learning parameters
y = .99
e = 0.4
num_episodes = 4000
# create lists to contain total rewards and steps per episode
jList = []
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    d = False
    j = 0
    # The Q-Network
    while j < 99:
        j += 1
        # 1. Choose an action greedily from the Q-network
        #    (run the network for current state and choose the action with the maxQ)
        hot_vector = torch.zeros(16)
        hot_vector[state] = 1
        Q = net(hot_vector)
        action = torch.argmax(Q).item()
        # 2. A chance of e to perform random action
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        # 3. Get new state(mark as s1) and reward(mark as r) from environment
        s1, r, d, _ = env.step(action)
        # 4. Obtain the Q'(mark as Q1) values by feeding the new state through our network
        hot_vector = torch.zeros(16)
        hot_vector[s1] = 1
        Q1 = net(hot_vector)

        # 5. Obtain maxQ' and set our target value for chosen action using the bellman equation.
        Q_target = Variable(Q.data)
        Q_target[action] = r + y * torch.max(Q1).item()

        # 6. Train the network using target and predicted Q values (model.zero(), forward, backward, optim.step)
        loss = criterion(Q_target, Q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rAll += r
        state = s1
        if d == True:
            #Reduce chance of random action as we train the model.
            e = 1./((i/50) + 10)
            break
    jList.append(j)
    rList.append(rAll)

# Reports
print("Score over time: " + str(sum(rList)/num_episodes))
