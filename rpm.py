# from collections import deque
import numpy as np
import random

import pickle

# replay buffer per http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
class RPM(object):
    #replay memory
    def __init__(self,buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0

        #used for scaling
        self.min_reward = 0.0
        self.max_reward = 1.0

    def store(self, obj):
        if self.size() > self.buffer_size:
            #trim
            print('buffer size larger than set value, trimming...')
            self.buffer = self.buffer[(self.size()-self.buffer_size):]

        elif self.size() == self.buffer_size:
            self.buffer[self.index] = obj
            self.index += 1
            self.index %= self.buffer_size

        else:
            self.buffer.append(obj)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size):
        '''
        batch_size specifies the number of experiences to add
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least
        batch_size elements before beginning to sample from it.
        '''

        if self.size() < batch_size:
            batch = random.sample(self.buffer, self.size())
        else:
            batch = random.sample(self.buffer, batch_size)

        item_count = len(batch[0])
        res = []
        for i in range(item_count):
            # k = np.array([item[i] for item in batch])
            k = np.stack((item[i] for item in batch),axis=0)
            # if len(k.shape)==1: k = k.reshape(k.shape+(1,))
            if len(k.shape)==1: k.shape+=(1,)
            res.append(k)
        return res
    def zero_one_scale(self, r_i):
        return (r_i - self.min_reward) / self.max_reward

    def save(self, pathname):
        reward_index = 2
        self.min_reward = min([r_i[reward_index] for r_i in self.buffer])
        # add self.min_reward b/c we pushed everything to the right. ex) new_r = (r-min)/max
        self.max_reward = max([r_i[reward_index] for r_i in self.buffer]) + self.min_reward
        pickle.dump(self, open(pathname, 'wb'))
        print('memory dumped into',pathname)
