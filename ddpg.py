import tensorflow as tf
from tensorflow.contrib.layers import layer_norm
import numpy as np
import gym
from noise import OUNoise
from rpm import RPM

#####################  hyper parameters  ####################

MAX_EPISODES = 100
MAX_EP_STEPS = 500
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.99     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 1000000
LEARN_START = 10000
BATCH_SIZE = 64

RENDER = False
ENV_NAME = 'Pendulum-v0'

###############################  DDPG  ####################################
def lrelu(x, leak=0.2):
   return tf.maximum(x, leak * x)

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.rpm = RPM(MEMORY_CAPACITY)
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
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        [bs, ba, br, bs_] = self.rpm.sample(BATCH_SIZE)
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

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

###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
a_low = env.action_space.low

ddpg = DDPG(a_dim, s_dim, a_bound)
ou_noise = OUNoise(a_dim, sigma=0.2)
var = 2  # control exploration
for i in range(MAX_EPISODES*10):
    s = env.reset()
    ep_reward = 0.0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        if i < MAX_EPISODES:
            add_noise = ou_noise.noise()
            a = a + add_noise    # add randomness to action selection for exploration
        a = np.clip(a, a_low, a_bound) 
        s_, r, done, info = env.step(a)
        if i < MAX_EPISODES:
            ddpg.rpm.store([s, a, r, s_])

            if ddpg.rpm.size() >= LEARN_START:
                ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %.3f' % ep_reward, 'Explore: {}'.format(add_noise))
