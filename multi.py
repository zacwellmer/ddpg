# FAST ENV
# this is a environment wrapper. it wraps the RunEnv and provide interface similar to it. The wrapper do a lot of pre and post processing (to make the RunEnv more trainable), so we don't have to do them in the main program.
import numpy as np

class fastenv:
    def __init__(self,e,skipcount):
        self.e = e
        self.stepcount = 0
        self.obs_dim = self.e.observation_space.shape[0]

        self.prev_obs = np.zeros(self.obs_dim)
        self.skipcount = skipcount

    def obg(self,plain_obs):
        # observation generator
        # derivatives of observations extracted here.
        plain_obs = np.zeros(self.obs_dim) if plain_obs is None else plain_obs
        processed_obs = np.hstack((self.prev_obs, plain_obs))
        return processed_obs

    def step(self,action):
        action = [float(action[i]) for i in range(len(action))]

        import math
        for num in action:
            if math.isnan(num):
                print('NaN met',action)
                raise RuntimeError('this is bullshit')

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
        self.prev_obs = np.zeros(self.obs_dim)

        oo = self.e.reset()
        o = self.obg(oo)
        return o
