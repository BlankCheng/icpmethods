# -*- coding: utf8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
import math
import random
from icp_point_to_point_SVD import *


# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
DIST_THRESHOLD = 1e-5
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100
NUM_ACTIONS = 6 # 0 1 2 3 分别代表R=-0.5; R=0.5; t=-0.05; t=0.05， 每次只选择一种策略
NUM_STATES = 10 * 3 # cloud size

#env = gym.make("CartPole-v0")
#env = env.unwrapped
#ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

# Net
class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob


# env
class Env():
    def __init__(self, cloudRead, cloudRefence):
        self.cloudRead = cloudRead
        self.cloudReference = cloudReference
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2)) # state is 100 * 3
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.action = 0
        self.state = self.cloudRead.reshape(1, -1) # 当前点云位置
        self.reward = 0
        self.prev_dist = 0 # 上一步的dist
        self.T = np.identity(4)

    def cal_reward(self):
        distances, indices = matchPoints(self.cloudRead, self.cloudReference)
        distances = np.mean(distances)
        self.prev_dist = distances
        delta_dist = self.prev_dist - distances
        self.reward = delta_dist * 10

        return self.reward

    def cal_action(self):

        self.state = torch.tensor(self.state).float()
        if np.random.randn() <= EPISILO:# greedy policy
            action_value = self.eval_net.forward(self.state)
            self.action = torch.max(action_value, 1)[1].data.numpy().item()
        else: # random policy
            self.action = np.random.randint(0,NUM_ACTIONS)

        return self.action

    def cal_state(self):
        a = 0
        b = 0
        c = 0
        x = 0
        y = 0
        z = 0
        print ("action: ",self.action)
        if self.action == 0:
            a = 0.5
        elif self.action == 1:
            b = 0.5
        elif self.action == 2:
            c = 0.5
        elif self.action == 3:
            x = 0.05
        elif self.action == 4:
            y = 0.05
        elif self.action == 5:
            z = 0.05

        T = generateTransformByEuler(a, b, c, x, y, z)
        self.cloudRead = cloudTransform(T, self.cloudRead)
        self.state = self.cloudRead.reshape(1, -1)
        self.T = np.dot(T, self.T)
        print (T)
        print (self.T)
        return self.state

    def reset_env(self):
        self.cloudRead = cloudRead
        self.cloudReference = cloudReference
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2)) # state is 100 * 3
        self.action = 0
        self.state = self.cloudRead.reshape(1, -1) # 当前点云位置
        self.reward = 0
        distances, _ = matchPoints(self.cloudRead, self.cloudReference)
        self.prev_dist = np.mean(distances)
        self.T = np.identity(4)

    def store_transition(self, state):
        #print (state)
        #print ([self.action, self.reward])
        #print (self.state)
        transition = np.hstack((state.squeeze(), [self.action, self.reward], self.state.squeeze()))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])

        #q_eval是以最近似地表达q为目的，而不是最优，最优的任务交给q
        #两个网络是因为有一个网络要保持不更新
        #网络是为了代替q_table, 可以直接计算出q
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1) # 只是之前的一种a=1的特殊情况
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




if __name__ == '__main__':
    Tg = generatePerturbation(30, 1)
    cloudRead = np.random.rand(10, 3)
    cloudReference = cloudTransform(Tg, cloudRead)
    env = Env(cloudRead, cloudReference)

    episodes = 400
    print("Collecting Experience....")
    reward_list = []
    plt.ion()
    fig, ax = plt.subplots()
    for i in range(episodes):
        env.reset_env()
        ep_reward = 0
        while True:
            action = env.cal_action()
            state = env.cal_state()
            reward = env.cal_reward()
            env.store_transition(state)
            ep_reward += reward
            if env.memory_counter >= MEMORY_CAPACITY:
                err_t, err_R = eval_err(env.T, Tg)
                print ("err_R:", err_R)
                print ("err_t:", err_t)
                print ("avg_dist:",env.prev_dist)
                env.learn()
            if env.prev_dist < DIST_THRESHOLD:
                print("episode:{}, the distance is {}, the ep_reward is {}".format(i, env.prev_dist, ep_reward ))
                print("transformation: ", env.T)
                break

        r = copy.copy(reward)
        reward_list.append(r)
        ax.set_xlim(0,300)
        #ax.cla()
        ax.plot(reward_list, 'g-', label='reward')
        plt.pause(0.001)




