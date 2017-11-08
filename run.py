import os
import sys
import numpy as np
import time
import pickle

from ddpg import DDPG
from multi import fastenv
import gym

if __name__=='__main__':
    #fast_e = fastenv(gym.make('BipedalWalker-v2'), skipcount=1)
    fast_e = fastenv(gym.make('Pendulum-v0'), skipcount=1)
    obs_space_dims = fast_e.e.observation_space.shape[0] * 2

    agent = DDPG(
    observation_space=obs_space_dims,
    action_space=fast_e.e.action_space,
    discount_factor=.995
    )
    def r(start_ep, ep,times=1):
        for i in range(start_ep, ep):
            if i % 10 == 0:
                print('ep',i+1,'/',ep,'times:',times)
                sys.stdout.flush()

            agent.play(fast_e,ep_i=i,max_steps=-1)

            time.sleep(0.05)

            if (i+1) % 100 == 0:
                # save the training result.
                save(i+1)

    def save(i):
        rpm_loc = 'rpm.pickle'
        print('saving rpm at {} to {}'.format(i, rpm_loc))
        agent.lock.acquire()
        agent.save_weights(i)
        agent.rpm.save('rpm.pickle')
        agent.lock.release()

    def load():
        start_ep = agent.load_weights()
        if os.path.exists('rpm.pickle'):
            agent.rpm=pickle.load(open('rpm.pickle', 'rb'))
        return start_ep

    start_ep = load()
    r(start_ep, 50000)
