# https://github.com/songrotek/DDPG/blob/master/ou_noise.py
import numpy as np
import numpy.random as nr
class AdditiveGaussian(object):
    def __init__(self, action_dimension, mu=0, sigma=0.2, decay=0.999):
        self.action_dimension = action_dimension
        self.mu = mu
        self.sigma = sigma
        self.decay = decay

    def reset(self):
        pass

    def noise(self, ep_i):
        scale = self.sigma * self.decay ** ep_i
        scale = 0.05 if scale < 0.05 else scale
        return np.random.normal(loc=self.mu, scale=scale, size=self.action_dimension)

class OUNoise(object):
    """docstring for OUNoise"""
    def __init__(self,action_dimension, mu=0, sigma=0.2, theta=.15):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.zeros(self.action_dimension)

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(size=self.action_dimension)
        self.state = x + dx
        return self.state

