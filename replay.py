from collections import defaultdict

import numpy as np
import torch


class Replay:
    def __init__(self, size, batch_size, device):
        self.MEMORY_CAPACITY = size
        self.memory_counter = 0
        self.BATCH_SIZE = batch_size
        self.cuda_info = device is not None

    def _sample(self):
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        return sample_index

    def sample(self):
        raise NotImplementedError()

    def store_transition(self, resource):
        raise NotImplementedError()


class RandomClusterReplay(Replay):
    def __init__(self, size, batch_size, state_shape, device, op_dim=0):
        super().__init__(size, batch_size, device)
        self.memory = np.zeros((self.MEMORY_CAPACITY, state_shape * 2 + state_shape *
                                2 + op_dim * 2 + 1))
        self.STATE_DIM = state_shape
        self.ACTION_DIM = op_dim
        if self.cuda_info:
            self.mem1 = self.mem1.cuda()
            self.mem2 = self.mem2.cuda()
            self.reward = self.reward.cuda()

    def store_transition(self, mems):
        s, a, r, s_, a_ = mems
        transition = np.hstack((s, a, [r], s_, a_))
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def sample(self):
        sample_index = self._sample()
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.STATE_DIM])
        b_a = torch.LongTensor(b_memory[:, self.STATE_DIM:self.STATE_DIM +
                                                          self.ACTION_DIM])
        b_r = torch.FloatTensor(b_memory[:, self.STATE_DIM + self.
                                ACTION_DIM:self.STATE_DIM + self.ACTION_DIM + 1])
        b_s_ = torch.FloatTensor(b_memory[:, self.STATE_DIM + self.
                                 ACTION_DIM + 1:self.STATE_DIM * 2 + self.ACTION_DIM + 1])
        b_a_ = torch.LongTensor(b_memory[:, -self.ACTION_DIM:])
        return b_s, b_a, b_r, b_s_, b_a_


class RandomOperationReplay(Replay):
    def __init__(self, size, batch_size, state_dim, device):
        super().__init__(size, batch_size, device)
        self.memory = np.zeros((self.MEMORY_CAPACITY, state_dim * 2 + 2))
        self.N_STATES = state_dim

    def store_transition(self, mems):
        s1, op, r, s2 = mems
        transition = np.hstack((s1, [op, r], s2))
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
        # self.mem1[index] = s1
        # self.mem2[index] = s2
        # self.reward[index] = r
        # self.op[index] = op
        # self.memory_counter += 1

    def sample(self):
        sample_index = self._sample()
        b_memory = self.memory[sample_index]
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES])
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES + 1])
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES + 1:self.N_STATES +
                                                              2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_STATES:])
        return b_s, b_a, b_r, b_s_


class PERClusterReplay(RandomClusterReplay):
    def __init__(self, size, state_dim, action_dim, batch_size):
        super().__init__(size, state_dim, action_dim, batch_size)

    def _sample(self):
        raise NotImplementedError()
