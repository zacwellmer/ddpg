from __future__ import print_function

# gym boilerplate
import time
import numpy as np
import sys
import gym
from gym import wrappers
from gym.spaces import Discrete, Box

from math import *
import random
import time

from rpm import rpm # replay memory implementation
from noise import OUNoise, AdditiveGaussian

import tensorflow as tf
from tensorflow.python import debug as tfdbg
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import layer_norm

import traceback

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

class NN(object):
    def __init__(self, netname, statedims, trainable=True):
        self.netname = netname
        self.trainable = trainable
        self.statedims = statedims
        self.common_hidden = [400, 300]

    # the part of network that the input and output shares architechture
    def build_common(self):
        with tf.variable_scope('common'):
            self.s = tf.placeholder(tf.float32, [None, self.statedims], name='s')
            normalized_s = layer_norm(self.s)            
            h1 = layer_norm(slim.fully_connected(normalized_s, self.common_hidden[0],
                            activation_fn=lrelu,
                            trainable=self.trainable))
            self.h2 = layer_norm(slim.fully_connected(h1, self.common_hidden[1],
                                 activation_fn=lrelu,
                                 trainable=self.trainable))

class Actor(NN):
    def __init__(self, netname, statedims, actiondims, trainable):
        super().__init__(netname, statedims, trainable=trainable)

        self.hidden_dims = [128]
        self.action_dims = actiondims
        self.build()

    def build(self):
        with tf.variable_scope(self.netname):
            self.build_common()
            a_h2 = layer_norm(slim.fully_connected(self.h2, self.hidden_dims[0],
                              activation_fn=lrelu,
                              trainable=self.trainable))
            self.a = slim.fully_connected(a_h2, self.action_dims,
                                         activation_fn=tf.nn.tanh,
                                         trainable=self.trainable)
            self.a = tf.identity(self.a, self.netname+'/a')

class Critic(NN):
    def __init__(self, netname, statedims, action_in, trainable):
        super().__init__(netname, statedims, trainable=trainable)
        self.hidden_dims = [128]
        self.action_in = action_in
        self.build()

    def build(self):
        with tf.variable_scope(self.netname):
            self.build_common()
            action_h2 = tf.concat([self.action_in, self.h2], axis=1)
            normalized_action_h2 = layer_norm(action_h2)
            h_out = layer_norm(slim.fully_connected(normalized_action_h2, self.hidden_dims[0],
                               activation_fn=lrelu,
                               weights_regularizer=slim.l2_regularizer(0.01),
                               trainable=self.trainable))
            self.Q = slim.fully_connected(h_out, 1, activation_fn=None,
                                     weights_regularizer=slim.l2_regularizer(0.01),
                                     trainable=self.trainable)

class DDPG(object):
    def __init__(self, observation_space, action_space,
                 discount_factor=0.99):
        self.rpm = rpm(1000000)
        self.training = True
        self.discount_factor = discount_factor

        self.inputdims = observation_space

        self.action_high, self.action_low = action_space.high, action_space.low
        self.actiondims = action_space.shape[0]
        self.exploration_noise = OUNoise(self.actiondims, sigma=0.2)
        #self.exploration_noise = AdditiveGaussian(self.actiondims, sigma=1.0)

        self.r1 = tf.placeholder(tf.float32,shape=[None,1], name='reward')
        self.ActorEval = Actor('actor_eval', self.inputdims, self.actiondims, trainable=True)
        self.ActorTarget = Actor('actor_target', self.inputdims, self.actiondims, trainable=False)
        self.CriticEval = Critic('critic_eval', self.inputdims, self.ActorEval.a, trainable=True)
        self.CriticTarget = Critic('critic_target', self.inputdims, self.ActorTarget.a, trainable=False)

        self.build()

        self.total_episode_reward = tf.placeholder(tf.float32, [], name='episode_reward')
        tf.summary.scalar('total_episode_reward', self.total_episode_reward)
        self.merged = tf.summary.merge_all()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.train_writer = tf.summary.FileWriter('tensorboard/', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        self.sync_target()

        import threading as th
        self.lock = th.Lock()

    def clamper(self, actions):
        return np.clip(np.nan_to_num(actions), a_max=self.action_high, a_min=self.action_low)

    def shift(self):
        # 3. shift the weights (aka target network)
        self.tau = tf.Variable(1e-3)
        ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_eval')
        at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_target')
        ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_eval')
        ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_target')

        self.actor_shift = [tf.assign(t, (1 - self.tau) * t + self.tau * e) for t, e in
zip(at_params, ae_params)]
        self.critic_shift = [tf.assign(t, (1 - self.tau) * t + self.tau * e) for t, e in
zip(ct_params, ce_params)]

    def build(self):
        # 1. update the critic
        q_target = self.r1 + self.discount_factor * self.CriticTarget.Q
        q_target = tf.identity(q_target, 'q_target')
        self.critic_loss = tf.losses.mean_squared_error(q_target, self.CriticEval.Q)
        self.critic_loss = tf.identity(self.critic_loss, 'critic_loss')
        # 2. collect TD errors for sampling
        self.td_errors = q_target - self.CriticEval.Q

        # 3. update the actor
        self.actor_loss = tf.reduce_mean(- self.CriticEval.Q)
        self.actor_loss = tf.identity(self.actor_loss, 'actor_loss')
        # maximize q1_predict -> better actor

        # 4. shift the weights (aka target network)
        self.shift()

        # optimizer on
        # actor is harder to stabilize...
        critic_opt = tf.train.AdamOptimizer(1e-3, name='adam_critic')
        self.critic_train = critic_opt.minimize(self.critic_loss)

        actor_opt = tf.train.AdamOptimizer(1e-4, name='adam_actor')
        a_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_eval')
        q_grads = [-1.0 * grad for grad in tf.gradients(self.CriticEval.Q, self.CriticEval.action_in)]
        a_grads = tf.gradients(self.ActorEval.a, a_params, q_grads)
        self.actor_train = actor_opt.apply_gradients(zip(a_grads, a_params))

        tf.summary.scalar('critic_loss_summary', tf.reduce_sum(self.critic_loss))
        tf.summary.scalar('actor_loss_summary', tf.reduce_sum(self.actor_loss))

    def feed_train(self, s1d, a1d, r1d, s2d):
        #update actor
        aloss, _ = self.sess.run([self.actor_loss, self.actor_train],
            feed_dict={
            self.CriticEval.s: s1d,
            self.ActorEval.s: s1d,
            })

        # update critic
        td_errors, closs, _= self.sess.run([self.td_errors, self.critic_loss,
                            self.critic_train],
            feed_dict={
            self.CriticEval.s: s1d,
            self.CriticEval.action_in: a1d,
            self.CriticTarget.s: s2d,
            self.ActorEval.s: s1d,
            self.ActorTarget.s: s2d,
            self.r1:r1d,
            })

        _, _ = self.sess.run([self.critic_shift, self.actor_shift], feed_dict={self.tau:1e-2})
        return aloss, closs, td_errors

    def joint_inference(self, state):
        a1d = self.sess.run(self.ActorEval.a, feed_dict={self.ActorEval.s: state})
        q1d = self.sess.run(self.CriticEval.Q, feed_dict={self.CriticEval.s: state,
                                                          self.CriticEval.action_in:a1d})
        return [a1d, q1d]

    def sync_target(self):
        self.sess.run([self.actor_shift, self.critic_shift],feed_dict={self.tau:1.})

    def train(self, ep_i):
        batch_size = 64
        epochs = 1
        aloss, closs = None, None # if rpm too small
        if self.rpm.size() > batch_size:
            #if enough samples in memory
            # sample randomly a minibatch from memory
            [s1d, a1d, r1d, s2d] = self.rpm.sample(batch_size)
            scaled_r1d = [self.rpm.zero_one_scale(r_i) for r_i in r1d.squeeze()] # scale rewards to be 0 - 1
            scaled_r1d = np.reshape(scaled_r1d, [-1, 1])
            aloss, closs, td_errors = self.feed_train(s1d, a1d, scaled_r1d, s2d)
        return aloss, closs

    # gymnastics
    def play(self,env, ep_i, max_steps=-1): # play 1 episode
        timer = time.time()

        max_steps = max_steps if max_steps > 0 else 50000
        steps = 0
        total_reward = 0
        episode_memory = []

        # removed: state stacking
        # moved: observation processing

        try:
            observation = env.reset()
        except Exception as e:
            print('(agent) something wrong on reset(). episode terminates now')
            traceback.print_exc()
            print(e)
            return

        while True and steps <= max_steps:
            steps +=1

            observation_before_action = observation # s1
            add_noise = self.exploration_noise.noise()
            action = self.act(observation_before_action) + add_noise # a1
            action = self.clamper(action)
            if np.random.uniform()<0.001: print('noise added: ', add_noise, 'action: ', action)
            action_out = action

            # o2, r1,
            try:
                observation, reward, done, _info = env.step(action_out) # take long time
            except Exception as e:
                print('(agent) something wrong on step(). episode teminates now')
                traceback.print_exc()
                print(e)
                print('Action:', action, 'observation: ', observation_before_action)
                return

            # d1
            total_reward += reward

            # feed into replay memory
            if self.training == True:
                episode_memory.append((
                    observation_before_action,action,reward,observation
                ))

                # don't feed here since you never know whether the episode will complete without error.
                aloss, closs = self.train(ep_i)

            if done :
                self.exploration_noise.reset()
                break

        totaltime = time.time()-timer
        summary = self.sess.run(self.merged,
                                feed_dict={self.total_episode_reward: total_reward,
                                self.actor_loss: aloss, self.critic_loss: closs})
        self.train_writer.add_summary(summary, ep_i)
        if ep_i % 10 == 0:
            print('episode done in {} steps in {:.2f} sec, {:.4f} sec/step, got reward :{:.2f}'.format(
        steps,totaltime,totaltime/steps,total_reward
        ))
        sys.stdout.flush()

        self.lock.acquire()
        for t in episode_memory:
            self.rpm.store(t)
        self.lock.release()

        return

    # one step of action, given observation
    def act(self,observation):
        obs = np.reshape(observation,(1,len(observation)))
        [actions,q] = self.joint_inference(obs)
        actions,q = actions[0],q[0]
        return actions

    def save_weights(self, i):
        self.saver.save(self.sess, 'checkpoints/ddpg', i)

    def load_weights(self):
        latest_loc = tf.train.latest_checkpoint('checkpoints')
        if latest_loc is not None:
            self.saver.restore(self.sess, latest_loc)
            return int(latest_loc.split('-')[-1]) # episode
        return 0
