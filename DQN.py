import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

# plt.figure('rc', )
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

import math


class ReplayMemory():

    def __init__(self, capacity):
        self.capacity = capacity
        self.mem = []
        self.ptr = 0

    def push(self, *args):
        """Save trans to memory"""
        new_trans = Transition(*args)
        if len(self.mem) < self.capacity:
            self.mem.append(new_trans)
        else:
            # Use random Replay memory
            self.mem[self.ptr] = new_trans
            self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size=128):
        # Random.sample is very useful
        return random.sample(self.mem, batch_size)

    def can_sample(self, batch_size=120):
        #         return batch_size < len(self.mem)
        return self.capacity == len(self.mem)


class DQNetState(nn.Module):

    def __init__(self, state, action_dims):
        super(DQNetState, self).__init__()
        self.fc1 = nn.Linear(state, 50)
        self.fc2 = nn.Linear(50, action_dims)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


import random


# DQN Algoritms for 4 input and  2 output
class DQNwithState:

    def __init__(self, state_dim, action_dim, env, batch_size=128, gamma=0.1, epsilon=0.3, lr=0.001, target_update=10):
        self.PolicyQNet = DQNetState(state_dim, action_dim)
        self.TargetQNet = DQNetState(state_dim, action_dim)


        self.state_dim = state_dim
        self.action_dim = action_dim

        self.env = env

        # self.reopen_env()
        self.step = 0
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.mem = ReplayMemory(100)
        #         self.optimizer = optim.RMSprop(self.PolicyQNet.parameters(),lr=lr)
        self.optimizer = optim.Adam(self.PolicyQNet.parameters(), lr=lr)

        self.epoch = 0
        self.lost = []

        self.reward_now = 0

        self.criterion = nn.MSELoss()
        self.total_reward = []
        self.reward_array = []

        self.target_update = target_update

        self.train_episode = 30*100

        self.Q_logs = []

        self.first_draw = 1


    def load_target_QNet(self):
        self.TargetQNet.load_state_dict(self.PolicyQNet.state_dict())

    def reopen_env(self):
        self.env.reset()

    def show_state(self):
        fig = plt.figure(3)
        plt.clf()
        # plt.imshow(self.env.render(mode='rgb_array'))
        plt.title(
            "reward %f Epoch :%d x %f th %f" % (self.reward_now, self.epoch, self.env.state()[0], self.env.state()[1]))
        # plt.axis('off')

        plt.subplot(311)
        plt.semilogy(self.lost, label="LOST")
        plt.grid(which="both")
        plt.legend()

        plt.subplot(312)
        plt.plot(self.reward_array, label="Reward")
        plt.legend()
        plt.grid(which="both")

        plt.subplot(313)
        plt.plot(self.total_reward, label="Total Reward")
        plt.legend()
        plt.grid(which="both")
        plt.pause(0.001)

    def found_max_Q(self, state):
        print(np.array(state))
        res = self.PolicyQNet(Variable(torch.Tensor(state)))
        print(res)
        print(res.max(0))
        return res.max(0)

    def egreedy_action(self, state):
        if random.random() > self.epsilon:
            return random.randint(0, 1)
        Q_last, act = self.found_max_Q(state)
        # print(Q_last.data.numpy())
        self.Q_logs.append(Q_last.data.numpy())
        return int(act.data.numpy())

    def train_batch(self):
        # Get batch
        if not self.mem.can_sample(self.batch_size):
            #             print("Train EPI {} : not enough data".format(episode),end="")
            return

        batches = self.mem.sample(self.batch_size)
        batches = Transition(*zip(*batches))

        # print("action",list(batches.action))
        state_batches = torch.Tensor(list(batches.state))
        action_batches = Variable(torch.LongTensor(list(batches.action))).view(self.batch_size, 1)
        next_state_batches = Variable(torch.Tensor(list(batches.next_state)))
        reward_batches = Variable(torch.Tensor(list(batches.reward)).view(self.batch_size, 1))
        next_Q_batches = self.TargetQNet(next_state_batches).detach()

        done_batches = list(batches.done)

        ys = reward_batches.data.numpy()
        ys2 = self.gamma * next_Q_batches.max(1)[0].view(self.batch_size, 1).data.numpy()

        def add_by_done(ys1, ys2, done):
            if done:
                return float(ys1[0])
            return float((ys1 + ys2)[0])

        ys = list(map(add_by_done, ys, ys2, done_batches))
        # print("Ys",ys)
        ys = Variable(torch.Tensor(ys))

        Qs = self.PolicyQNet(Variable(state_batches))
        Qs = Qs.gather(1, action_batches)
        loss = self.criterion(Qs, ys)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.PolicyQNet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.lost.append(loss.data)

        print("", end="\r")
        print("Epoch {} Trained {} reward {} loss {}".format(self.epoch, self.step, self.reward_now, loss.data.numpy()),
              end="")

    def train_loop(self, show_debug=False, epi=None):
        if epi is not None:
            self.epsilon = epi

        state_last = self.env.state()
        total_reward = 0
        for episode in range(self.train_episode):
            #         while True:
            self.step = self.step + 1
            if show_debug and episode % 100 == 0:
                self.show_state()
            action = self.egreedy_action(state_last)

            state, reward, done, _ = self.env.step(action)
            self.reward_array.append(reward)
            # print(reward)
            state = list(state)

            # reward = self.reward()

            self.mem.push(state_last, action, state, reward, done)
            state_last = state
            self.reward_now = reward
            self.train_batch()
            total_reward = total_reward + reward
            if self.step % self.target_update == 0:
                self.load_target_QNet()
            if done:
                self.total_reward.append(total_reward)
                return
        self.total_reward.append(total_reward)

    def test_loop(self, EPISODE=10000, show_debug=True):
        self.env.reset()
        state_last = [0, 0, 0, 0]
        total_reward = 0
        for episode in range(EPISODE):
            if show_debug:
                self.show_state()
            Qmaxovera, act = self.found_max_Q(state_last)
            action = int(act.data.numpy())
            state, reward, done, _ = self.env.step(action)

            reward = self.reward()

            if done:
                return total_reward, episode

            state_last = list(state)
            total_reward = total_reward + reward

        return total_reward, episode

    def test(self, EPISODE=10000):
        reward_list = []
        episode_list = []
        for i in range(12):
            reward, episode = self.test_loop(EPISODE=EPISODE, show_debug=False)
            reward_list.append(reward)
            episode_list.append(episode)
        return np.mean(reward_list), np.mean(episode_list)

    def main_loop(self, epoch=100, show_debug=False, init_epi=1.0):
        print("Start DQN Train")
        init_epi = 1.0
        for i in range(epoch):
            self.env.reset()
            self.epoch = i
            epi = init_epi * math.exp(0.1 * -i / epoch)
            epi = 0.95
            self.train_loop(show_debug=show_debug, epi=epi)


# DQN = DQNwithState(batch_size=4)
# DQN.main_loop(10)
# DQN.env.close()

# except Exception as inst:
#     print(inst)
#     DQN.env.close()