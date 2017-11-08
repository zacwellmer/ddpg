import os
import sys
import numpy as np
import time
import pickle

from ddpg import DDPG
from multi import fastenv
import gym

if __name__=='__main__':
    env_name = 'Pendulum-v0'
    rpm_loc = '{}-rpm.pickle'.format(env_name)
    checkpoint_dir = '{}-checkpoints/'.format(env_name)
    fast_e = fastenv(gym.make(env_name), skipcount=1)
    obs_space_dims = fast_e.e.observation_space.shape[0] * 2

    agent = DDPG(
    observation_space=obs_space_dims,
    action_space=fast_e.e.action_space,
    discount_factor=.995
    )
    def r(start_ep, ep,times=1):
        max_steps = 1000

        for i in range(start_ep, ep):
            if i % 10 == 0:
                print('ep',i+1,'/',ep,'times:',times)
                sys.stdout.flush()
            if i % 20:
                agent.play(fast_e,ep_i=i,max_steps=max_steps)
            else: # test performance
                agent.play(fast_e, ep_i=i, max_steps=max_steps,is_test=True)

            if (i+1) % 100 == 0:
                # save the training result.
                save(i+1)

    def save(i):
        print('saving rpm at {} to {}'.format(i, rpm_loc))
        agent.lock.acquire()
        agent.save_weights(checkpoint_dir, i)
        agent.rpm.save(rpm_loc)
        agent.lock.release()

    def load():
        start_ep = agent.load_weights(checkpoint_dir)
        if os.path.exists(rpm_loc):
            agent.rpm=pickle.load(open(rpm_loc, 'rb'))
        return start_ep

    start_ep = load()
    r(start_ep, 10000)
