import numpy as np
from collections import deque

class fastenv:
    def __init__(self, e, skipcount):
        self.e = e
        self.stepcount = 0
        self.obs_dim = self.e.observation_space.shape[0]
        self.skipcount = skipcount
        self.initialize_obs()
        self.a_dim = self.e.action_space.shape[0]
        self.s_dim = self.e.observation_space.shape[0] * self.skipcount

    def initialize_obs(self):
        self.prev_obs = deque(tuple(np.zeros((self.skipcount, self.obs_dim))), self.skipcount)

    def obg(self,plain_obs):
        # observation generator
        plain_obs = np.zeros(self.obs_dim) if plain_obs is None else plain_obs
        self.prev_obs.append(plain_obs)
        return np.array(self.prev_obs).flatten()

    def step(self,action):
        sr = 0
        for j in range(self.skipcount): #skip frames
            self.stepcount+=1
            oo,r,d,i = self.e.step(action)
            o = self.obg(oo)
            sr += r
            if d == True:
                break
        return o,sr,d,i

    def reset(self):
        self.stepcount=0
        self.initialize_obs()
        oo = self.e.reset()
        o = self.obg(oo)
        return o

