import tensorflow as tf
from tensorflow.contrib.layers import layer_norm
import numpy as np
import gym
import os
import sys
import pickle

from noise import OUNoise
from rpm import RPM

#####################  hyper parameters  ####################

MAX_EPISODES = 10000
MAX_EP_STEPS = 1000
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.99     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 1000000
LEARN_START = 16000
BATCH_SIZE = 64

RENDER = False
ENV_NAME = 'BipedalWalker-v2'#'Pendulum-v0'

###############################  DDPG  ####################################
def lrelu(x, leak=0.2):
   return tf.maximum(x, leak * x)

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.rpm = RPM({'size': MEMORY_CAPACITY, 'batch_size': BATCH_SIZE})
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params,
                             self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        self.w_is = tf.placeholder(tf.float32, shape=[None, 1], name='weighted_is')
        weighted_is = tf.unstack(self.w_is, num=BATCH_SIZE)

        self.td_errors = (q_target - q)**2
        self.td_errors = tf.unstack(self.td_errors, num=BATCH_SIZE)

        c_opt = tf.train.AdamOptimizer(LR_C)
        c_grads = [tf.gradients(td_i, self.ce_params) for td_i in self.td_errors]
        c_grads = list(map(lambda w_is_i, c_g: [w_is_i * c_g_i for c_g_i in c_g], weighted_is, c_grads))
        c_grads_agg = [0.0]*len(c_grads[0])
        for c_grad_i in c_grads:
            c_grads_agg = [c_grad_i[j]/BATCH_SIZE + c_grads_agg[j] for j in range(len(c_grads_agg))]
        self.ctrain = c_opt.apply_gradients(zip(c_grads_agg, self.ce_params))

        a_loss = - q    # maximize the q
        a_loss = tf.unstack(a_loss, num=BATCH_SIZE)

        a_opt = tf.train.AdamOptimizer(LR_A)
        a_grads = [tf.gradients(a_loss_i, self.ae_params) for a_loss_i in a_loss]
        a_grads = list(map(lambda w_is_i, a_g: [w_is_i * a_g_i for a_g_i in a_g], weighted_is, a_grads))
        a_grads_agg = [0.0]*len(a_grads[0])
        for a_grad_i in a_grads:
            a_grads_agg = [a_grad_i[j]/BATCH_SIZE + a_grads_agg[j] for j in range(len(a_grads_agg))]
        self.atrain = a_opt.apply_gradients(zip(a_grads_agg, self.ae_params))

        # logging
        self.episode_reward = tf.placeholder(tf.float32,name='epsiode_reward')
        tf.summary.scalar('episode_reward', self.episode_reward)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('/home/ubuntu/tensorboard/PER/', self.sess.graph)
        
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self, step_i):
        # soft target replacement
        self.sess.run(self.soft_replace)

        sample, w_is, e_id = self.rpm.sample(step_i)
        [bs, ba, br, bs_] = sample
        self.sess.run(self.atrain, {self.S: bs, self.w_is: np.reshape(w_is, [-1, 1])})
        _, td_errors = self.sess.run([self.ctrain, self.td_errors], {self.S: bs, 
                                        self.a: ba, self.R: br, self.S_: bs_, 
                                        self.w_is: np.reshape(w_is, [-1, 1])})
        self.rpm.update_priority(e_id, td_errors)
        if np.random.random() < 1.0/1.0e4: self.rpm.rebalance() # don't rebalanceoften

    def _build_a(self, s, scope, trainable):
        hidden_units = 60
        with tf.variable_scope(scope):
            #s = layer_norm(s)
            net = tf.layers.dense(s, hidden_units, activation=tf.nn.relu, name='l1', trainable=trainable)
            #net = layer_norm(net)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):        
        hidden_units = 60
        with tf.variable_scope(scope):
            #s = layer_norm(s)
            s_a = tf.concat([s, a], axis=1)
            #s_a = layer_norm(s_a)
            l1 = tf.layers.dense(s_a, hidden_units, activation=tf.nn.relu, name='l1', trainable=trainable) 
            #l1 = layer_norm(l1)
            return tf.layers.dense(l1, 1, trainable=trainable)  # Q(s,a)

    def write_reward(self, r, ep_i):
        summary = self.sess.run(self.merged,
                                feed_dict={self.episode_reward: r})
        self.train_writer.add_summary(summary, ep_i)

    def save(self, i):
        self.rpm.save('rpm.pickle')
        self.saver.save(self.sess, 'checkpoints/ddpg', i)

    def load(self):
        if os.path.exists('rpm.pickle'):
            pickle.load(open('rpm.pickle', 'rb'))

        latest_loc = tf.train.latest_checkpoint('checkpoints')
        last_ep = 0
        if latest_loc is not None:
            self.saver.restore(self.sess, latest_loc)
            last_ep = int(latest_loc.split('-')[-1])
        return last_ep
 
###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
a_low = env.action_space.low

ddpg = DDPG(a_dim, s_dim, a_bound)
start_ep = ddpg.load()
ou_noise = OUNoise(a_dim, sigma=0.2)
var = 2  # control exploration
for i in range(start_ep, MAX_EPISODES):
    is_test = bool(i % 20 == 0)
    s = env.reset()
    ep_reward = 0.0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        add_noise = np.zeros(a.shape) if is_test else ou_noise.noise()
        a = a + add_noise    # add randomness to action selection for exploration
        a = np.clip(a, a_low, a_bound) 
        s_, r, done, info = env.step(a)
        if not is_test:
            default_td_error = 100.0
            ddpg.rpm.store([s, a, r, s_, default_td_error])

            if ddpg.rpm.record_size >= LEARN_START:
                ddpg.learn(j)

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %.3f' % ep_reward, 'Explore: {}'.format(add_noise))
            sys.stdout.flush()

    if is_test: 
        ddpg.write_reward(ep_reward, i)
    if (i+1) % 100 == 0: # +1 so 0th iter is skipped
        ddpg.save(i+1)

