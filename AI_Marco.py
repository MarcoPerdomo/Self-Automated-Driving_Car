# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# creating the architecture of the Neural Network

class Network(nn.Module): #Inheriting from Module Parent class
    
    def __init__(self, input_size, nb_action): #INitialising the standar function
        super(Network, self).__init__() #Super funcion to use the modules of the Module parent class
        self.input_size = input_size #Creating an object from the Network class and specifying it gets the same value of the argument input size
        self.nb_action = nb_action #nb action will be equal to 3 for the 3 actions possible to take 
        self.fc1 = nn.Linear(input_size, 30) #Creating full connection to the first hidden layer
        self.fc2 = nn.Linear(30, nb_action) # Creating full connection to the output layer
        
    def forward(self, state): #Forward function to activate the neurons and return the Q values depending on the state
        x = F.relu(self.fc1(state)) #Activating hidden neurons by introducing the input states
        q_values = self.fc2(x) #Obtaining Q values in the output layer
        return q_values
    

# Implementing experience replay

class ReplayMemory(object): # Creating the replay experience class, it is similar to an LSTM
    
    def __init__(self, capacity): # Capacity of how many transitions it will learn from
        self.capacity = capacity # attaching the capacity to the future instance or object created
        self.memory = [] #Initialising the memory as a list 
        
    def push(self, event): #Creating the func that will append new events in the memory and make sure that memory does not exceed the capacity
        self.memory.append(event) #Appending the memory with the new events. Event will a tuple of 4 elements (last state, new state, last action, last reward)
        if len(self.memory) > self.capacity: 
            del self.memory[0] #Deleting the oldest event in the memory once reached the capacity
    
    def sample(self, batch_size): #Sample function will sample different batches
        samples = zip(*random.sample(self.memory, batch_size)) # random samples from memory that have a fixed size of batch_size. Zip is a reshape function
        return map(lambda x: Variable(torch.cat(x, 0)), samples) #map does the mapping from the samples to the Torch variables that includes tensor and gradient)
        # Lambda function takes the samples, concatenates with respect to their first dimension to be aligned. Variable converst into Torch variables


# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma # Gamma parameter
        self.reward_window = [] #Initialising the reward list that will be appended with all the rewards
        self.model = Network(input_size, nb_action) #calling the neural network
        self.memory = ReplayMemory(100000) # Only one argument, we choose the capacity of the memory
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001) #Arguments of the Adam class
        self.last_state = torch.Tensor(input_size).unsqueeze(0) # torch variable with an extra batch dimension
        self.last_action = 0 # Actions are indexes 0,1,2. If action is index 0, no rotation. Action index 1 is rotate right. Action index 2, rotate left
        self.last_reward = 0 # Initialising the reward to 0
        
    def select_action(self, state): #Calling the state (tuple of 5 arguments, 3 sensors, orientation and -orientation) to feed the neural network and obtain the output
        probs = F.softmax(self.model(Variable(state, volatile = True))*10) #Applying softmax to the output of the NN. Do not include the gradient. 
        # Temperature parameter = 7 that increases the certainty of the best action to take
        action = probs.multinomial(num_samples = 1) #random draw from the distribution probs
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1) #Output of the NN fed with batch state. We want the action that is chosen. 
        # Squeezing the output by killing the action and goign back to the simple form of the batches
        next_outputs = self.model(batch_next_state).detach().max(1)[0] #Getting the maximum of the q values on the next state represented by index 0, according to all the actions in index 1
        target = self.gamma*next_outputs + batch_reward #Target function 
        td_loss = F.smooth_l1_loss(outputs, target) #Obtaining the temporal difference, aka loss
        self.optimizer.zero_grad() #Reinitializing the optimizer at each iteration of the loop
        td_loss.backward(retain_graph = True) # improving the training performance with retaininb variables
        self.optimizer.step() #Backpropagates the error and uses the optimizer to update the weights
    
    def update(self, reward, new_signal): #Makes the connection between the AI and the game. Update the action, last state, last reward as soon as the AI selects an action
        new_state = torch.Tensor(new_signal).float().unsqueeze(0) #state is the new signal composed of 5 elements.
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state) #playing the action according to the new state 
        if len(self.memory.memory) > 100: #first memory is the object of the ReplayMemory class, second memory is the attribute of that class
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100) #Sampling 100 samples
            self.learn(batch_state, batch_next_state, batch_reward, batch_action) #learning from the new sample
        self.last_action = action #updating the action
        self.last_state = new_state #updating the state
        self.last_reward = reward #updating the reward
        self.reward_window.append(reward) # updating the reward window 
        if len(self.reward_window) > 1000:
            del self.reward_window[0] #Fixing the window size to 1000
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.) #mean of all the rewards in the reward window (avoiding it equal to 0)
    
    def save(self):#Saving two thigns: self.model and the optimizer
        torch.save({'state_dict': self.model.state_dict(), #saving the parameters as a dictionary. The dictionary needs Key and definitions
                    'optimizer': self.optimizer.state_dict(),
                    }, 'Last_brain.pth') #Saving in that file
    
    def load(self):
        if os.path.isfile('last_brain.pth'): #making sure there is a save file
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")        
