# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:43:47 2020

@author: Chatree
"""

#importing library
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#importing package from openAI and doom
import gym
from gym.wrappers import SkipWrapper
from ppaquetta_gym_doom.wrappers.action_space import ToDiscrete

#importing other libraries
import experience_reply,image_preprocessing

#BUILDING THE AI

#making the brain
class CNN(nn.Module):
    def __init__(self,number_actions):
        self.convolution1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5)
        self.convolution2=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3)
        self.convolution2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=2)
        self.fc1=nn.Linear(in_features=self.count_nuerons((1,80,80)),out_features=40)
        self.fc1=nn.Linear(in_features=40,out_features=number_actions)
        
    def count_nuerons(self,image_dim):
        x=Variable(torch.rand(1,*image_dim))
        x=F.relu(F.max_pool2d(self.convolution1(x),3,2))
        x=F.relu(F.max_pool2d(self.convolution2(x),3,2))
        x=F.relu(F.max_pool2d(self.convolution3(x),3,2))
        return x.data.view(1,-1).size(1)
    
    def forword(self,x):
        x=F.relu(F.max_pool2d(self.convolution1(x),3,2))
        x=F.relu(F.max_pool2d(self.convolution2(x),3,2))
        x=F.relu(F.max_pool2d(self.convolution3(x),3,2))
        x=x.view(x.size(0),-1)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x
    
#making the body
class SoftmaxBody(nn.Module):
    def __init__(self,T):
        super(SoftmaxBody,self).__init__()
        self.T=T
        
    def forward(self,outputs):
        probs=F.softmax(outputs*self.T)
        actions=probs.multinomial()
        return actions
        
#making the ai
class AI:
    def __init__(self,brain,body):
        self.brain=brain
        self.body=body
        
    def __call__(self,inputs):
        input=Variable(torch.from_numpy(np.array(inputs ,dtype=np.float32)))
        output=self.brain(input)
        actions=self.body(output)
        return actions.data.numpy()
        
    
# Part 2 - Training the AI with Deep Convolutional Q-Learning

# Getting the Doom environment
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)
number_actions = doom_env.action_space.n

#building the cnn
cnn=CNN(number_actions)
softmax_body=SoftmaxBody(T = 1.0)
ai=AI(cnn,softmax_body)

#implementing experience rply
n_steps=experience_reply.NStepProgress(env=doom_env,ai=ai,n_step=10)
memory=experience_reply.ReplayMemory(n_steps=n_steps,capacity = 10000)

#importing Eligibility criteria

def eligibility_trace(batch):
    gamma=0.99
    inputs=[]
    targets=[]
    for series in batch:
        input=Variable(torch.from_numpy(np.array([series[0].state,series[-1].state],dtype=np.float32)))
        output=cnn(input)
        cumu1_reward= 0.0 if series[-1].done else output[1].data.max
        for step in reversed(series[:-1]):
            cumu1_reward=step.reward + gamma*cumu1_reward
        state=series[0].state
        target=output[0].data
        target[series[0].action]=cumu1_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype=np.float32)),torch.stack(targets)


#making the moving average on 100 steps

class MA:
    def __init__(self,size):
        self.list_of_reward=[]
        self.size=size
        
    def add(self,rewards):
        if isinstance(rewards,list):
            self.list_of_rewards+=rewards
        else:
            self.list_of_rewards.append(rewards)
        while(len(self.list_of_rewards)> self.size):
            del self.list_of_rewards[0]
            
    def average(self):
        return np.mean(self.list_of_rewards)
ma=MA(100)
        
#training the AI

loss=nn.MSELoss()
optimizer=optim.Adam(cnn.parameters(),lr=0.001)
nb_epochs=100
for epochs in range(1,nb_epochs+1):
    memory.run_steps(200)
    for batch in memory.sample_batch(128):
        inputs,targets=eligibility_trace(batch)
        inputs,targets=Variable(inputs),Variable(targets)
        predictions=cnn(inputs)
        loss_error=loss(predictions,targets)
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()
        
    rewards_steps=n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward=ma.average()
    print("Epochs:%s ,Average_reward:%s" %(str(epochs),str(avg_reward)))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
