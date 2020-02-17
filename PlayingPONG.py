#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:00:37 2020
@author: alain
"""

import pong_utils
from parallelEnv import parallelEnv

import gym
import time

import numpy as np
import matplotlib.pyplot as plt
from model import Policy

import torch
import torch.optim as optim

algorithm = 'PPO'
LR = 1e-3
GAMMA = .995
BETA = .01
#for PPO
EPSILON = 0.1
SGD_epoch = 4


def get_future_rewards(rewards,num_trajectories,gamma =0.995):
    rewards = np.asarray(rewards)
    discounts = [gamma**i for i in range(len(rewards))]
    rewards_future = []
    for i in range(len(rewards)):
        if i == 0:
            current_discounts = discounts[:]
        else:
            current_discounts = discounts[:-i]
        R_list = []
        for j in range(num_trajectories):
            rew_agent = np.multiply(rewards[i:,j],current_discounts)
            R_list.append(np.sum(rew_agent))
        rewards_future.append(R_list)
    return np.asarray(rewards_future)

def get_future_rewards_recursive(rewards,gamma=0.995):
    rewards_discounted = []

    idx = range(0,len(rewards))
    Gt = sum([(gamma**ii)*rewards[ii] for ii in idx])
    
    rewards_discounted.append(Gt) # for the first visit, R0
    for i in range(1,len(rewards)):
        Gt = (Gt - rewards[i])/gamma 
        rewards_discounted.append(Gt)
    return np.asarray(rewards_discounted)

def reward_normalization(rewards):
    # to normalize between trajectories in each step --> step1 normalization,step2 normalization...
    #r_mean = np.mean(rewards_future, axis = 1)
    #r_std = np.std(rewards_future, axis = 1) + 1.0e-10
    #R_np = (rewards_future - r_mean[:,np.newaxis])/r_std[:,np.newaxis] #normalized
    r_mean = np.mean(rewards)
    r_std = np.std(rewards) + 1.0e-10
    return (rewards - r_mean)/r_std

def plotLearningCurve(mean_rewards):
    compact_scores = []
    for i in range(1,len(mean_rewards)+1):
        if i < 100:
            compact_scores.append(np.mean(mean_rewards[0:i]))
        else:
            compact_scores.append(np.mean(mean_rewards[i-100:i]))
    plt.plot(compact_scores)
    plt.title('Avg score over 100 consecutive episodes')
    plt.xlabel('# episode')
    plt.ylabel('Score')
    plt.show()
    
    plt.plot(mean_rewards)
    plt.xlabel('# episode')
    plt.ylabel('Score')
    plt.show()

def calculate_loss_REINFORCE(policy, old_probs, states, actions, rewards,
              gamma = 0.995, beta=0.01):

    ########
    ## CREDIT ASSIGNMENT --> TAKING INTO ACCOUNT ONLY FUTURE REWARDS
    ##                   --> DISCOUNTED REWARD IMPLEMENTED WITH GAMMA
    ## NOISE REDUCTION   --> NORMALIZATION OF REWARD 
    ########
    
    # get number of trajectories = num of agents
    steps_in_trajectories = len(states) 
    num_trajectories = len(states[0]) 
    
    actions = torch.tensor(actions, dtype=torch.int8, device=device) # ACTIONS: 'RIGHTFIRE' = 4 and 'LEFTFIRE" = 5
    new_probs = pong_utils.states_to_prob(policy, states) # convert states to policy (or probability)
    new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0-new_probs)
    
    # REWARDS
    #rewards_future = get_future_rewards_recursive(rewards,gamma)
    rewards_future = get_future_rewards(rewards,num_trajectories,gamma) 
    R_np = reward_normalization(rewards_future)
    with torch.no_grad():
        R = torch.from_numpy(R_np).float().to(device) #specify to prepare data for CUDA
    
    # POLICY_LOSS
    policy_loss = []
    for i,prob in enumerate(new_probs):
        log_prob = torch.log(prob)
        result = torch.mul(log_prob,R[i]).to(device) # CALCULATE GRADIENT --> multiply element-wise
        policy_loss.append(result)
    policy_loss = torch.cat(policy_loss) #concat in single 1D-tensor
    policy_loss = policy_loss.sum(dim = 0) # sum all values
    policy_loss /= num_trajectories # calculate the gradient estimation (divide total number of trajectories)
    return policy_loss

def clipped_surrogate_PPO(policy, old_probs, states, actions, rewards,
                      gamma = 0.995, epsilon=0.1, beta=0.01):

    # get number of trajectories = num of agents
    steps_in_trajectories = len(states) 
    num_trajectories = len(states[0]) 
    
    actions = torch.tensor(actions, dtype=torch.int8, device=device)

    # convert states to policy (or probability)
    new_probs = pong_utils.states_to_prob(policy, states)
    new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0-new_probs)
    
    rewards_future = get_future_rewards(rewards,num_trajectories,gamma)
    R = torch.tensor(reward_normalization(rewards_future)).float().to(device)
    
    # ratio for clipping
    old_probs = torch.tensor(old_probs).to(device)
    ratio = new_probs/old_probs

    # clipped function
    clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    clipped_surrogate = torch.min(ratio*R, clip*R)
    return torch.sum(clipped_surrogate)/num_trajectories
    """
    # include a regularization term
    # this steers new_policy towards 0.5
    # prevents policy to become exactly 0 or 1 helps exploration
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))
    
    #return torch.mean(beta*entropy + clipped_surrogate)
    """


# =============================================================================
# IMPORT ENVIRONMENT & SET DEVICE
# =============================================================================
device = pong_utils.device
print("using device: ",device)

# PongDeterministic does not contain random frameskip, so is faster to train than the vanilla Pong-v4 environment
env = gym.make('PongDeterministic-v4')
print("List of available actions: ", env.unwrapped.get_action_meanings())
# =============================================================================
# NEURAL NETWORK/ POLICY DEFINITION
# =============================================================================    
#policy=Policy().to(device)
policy=pong_utils.Policy().to(device)
print(policy)
optimizer = optim.Adam(policy.parameters(), lr=LR)
# =============================================================================
# PREPROCESSING EXAMPLE
# =============================================================================
# show what a preprocessed image looks like
env.reset()
_, _, _, _ = env.step(0)
# get a frame after 20 steps
for _ in range(20):
    frame, _, _, _ = env.step(1)

plt.subplot(1,2,1)
plt.imshow(frame)
plt.title('original image')

plt.subplot(1,2,2)
plt.title('preprocessed image')

# 80 x 80 black and white image
plt.imshow(pong_utils.preprocess_single(frame), cmap='Greys')
plt.show()
# =============================================================================
# ROLLOUT EXAMPLE
# =============================================================================
envs = pong_utils.parallelEnv('PongDeterministic-v4', n=4, seed=12345)
prob, state, action, reward = pong_utils.collect_trajectories(envs, policy, tmax=100)

# =============================================================================
# MAIN -- REINFORCE OR PPO
# =============================================================================
# training loop max iterations
episode = 500
tmax = 320

# widget bar to display progress
import progressbar as pb
widget = ['\ntraining loop: ', pb.Percentage(), ' ', 
          pb.Bar(), ' ', pb.ETA() ]
timer = pb.ProgressBar(widgets=widget, maxval=episode).start()

# initialize environment
envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)
# keep track of progress
mean_rewards = []

for e in range(episode):

    # collect trajectories
    old_probs, states, actions, rewards = pong_utils.collect_trajectories(envs, policy, tmax=tmax)
    total_rewards = np.sum(rewards, axis=0)
    
    if algorithm == 'REINFORCE':
        L = -calculate_loss_REINFORCE(policy, old_probs, states, actions, rewards, gamma=GAMMA, beta=BETA)
        #L = -pong_utils.surrogate(policy, old_probs, states, actions, rewards, beta=beta) # this is the SOLUTION!
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        del L
    else:
        # gradient ascent step
        for _ in range(SGD_epoch):
            # uncomment to utilize your own clipped function!
            L = -clipped_surrogate_PPO(policy, old_probs, states, actions, rewards, gamma=GAMMA, epsilon=EPSILON, beta=BETA)
            #L = -pong_utils.clipped_surrogate(policy, old_probs, states, actions, rewards, epsilon=epsilon, beta=beta)
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            del L
        # the clipping parameter reduces as time goes on
        EPSILON*=.999   
    # the regulation term also reduces
    # this reduces exploration in later runs
    BETA*=.995
    
    # get the average reward of the parallel environments
    mean_rewards.append(np.mean(total_rewards))
    
    # display some progress every 20 iterations
    if (e+1)%20 ==0 :
        print("\nEpisode: {0:d}, score: {1:f}".format(e+1,np.mean(total_rewards)))
        print(total_rewards)
    
    # update progress widget bar
    timer.update(e+1)
timer.finish()

plotLearningCurve(mean_rewards)
torch.save(policy, algorithm + '.pth')
