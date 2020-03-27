# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 18:38:31 2019

This method is from "Deep Reinforcement Learning in Large Discrete Action Space"
It combines three important sections which are DDPG, Ornstien-Uhlenbeck process and Wolpertinger.
1. DDPG is modified by the MorvanZhou github https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/9_Deep_Deterministic_Policy_Gradient_DDPG
2. Ornstein-Uhlenbeck process reference: https://github.com/rll/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
3. Utilize the neareset neighnors method following the paper to discrete the continuous action within space. 
  
@author: Chih Kai Cheng
"""

import tensorflow as tf
import numpy as np
from sklearn.neighbors import KDTree
import datetime, os, time


    
class DDPG(object):
    def __init__(self,
                 s_dim, a_dim,
                 lr_a=0.001, lr_c=0.01, gamma=0.9, tau=0.01,
                 memory_capacity=500000, batch_size=64,
                 graph=True, save=True):
        # restart the tensorflow static graph
        tf.reset_default_graph()
        # set the tensorflow random seed 
        tf.set_random_seed(2)
        # set the numpy random seed
        np.random.seed(2)
        self.a_dim, self.s_dim = a_dim, s_dim
        self.memory_capacity, self.batch_size = memory_capacity, batch_size
        self.memory = np.zeros((memory_capacity, s_dim * 2 + a_dim * 2), dtype=np.float32)

        self.pointer = 0
        # set the placeholder to put the data
        self.S  = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R  = tf.placeholder(tf.float32, [None, 1], 'r')
        
        self.sess = tf.Session()
        
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            self.q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            self.q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')
        
        
        #  soft replacement
        self.soft_replace = [tf.assign(t, (1 - tau) * t + tau * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]
        
        with tf.variable_scope("td_error_and_action_loss"):
            self.a_loss = - tf.reduce_mean(self.q,name="a_loss")    # maximize the q
            q_target = self.R + gamma * self.q_
            # in the feed_dic for the td_error, the self.a should change to actions in memory    
            self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)

        with tf.variable_scope("Train"):
            self.atrain = tf.train.AdamOptimizer(lr_a).minimize(self.a_loss, var_list=self.ae_params, name="atrain")
            self.ctrain = tf.train.AdamOptimizer(lr_c).minimize(self.td_error, var_list=self.ce_params, name="ctrain")

        #  Initailize the global variables 
        self.sess.run(tf.global_variables_initializer())
        if save:
            self.saver = tf.train.Saver(var_list = tf.trainable_variables())
        if graph:
            tf.summary.FileWriter('log/', self.sess.graph)
            
    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
    
    def evaluate_action(self, s, a):
        q = self.sess.run([self.q], {self.S:s, self.a:a})[0]
        q_sum = np.sum(q, axis=1)
        return a[np.argmax(q_sum), :]
        
    def learn(self):
        # sample data from the replay buffer
        indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        batch_memory = self.memory[indices, :]
        batch_s = batch_memory[:, :self.s_dim]
        batch_a = batch_memory[:, self.s_dim: self.s_dim + self.a_dim]
        batch_r = batch_memory[:, -self.s_dim - 1: -self.s_dim]
        batch_s_ = batch_memory[:, -self.s_dim:]
        
        # actor learning and critic learning
        self.sess.run([self.ctrain], {self.S: batch_s, self.a: batch_a, self.R: batch_r, self.S_: batch_s_})
        self.sess.run([self.atrain], {self.S: batch_s})

        # soft target replacement
        self.sess.run(self.soft_replace)
        # retreive the loss and the error to the monitor
        self.loss = self.a_loss.eval(session=self.sess, feed_dict={self.S: batch_s})
        self.error = self.td_error.eval(session=self.sess, feed_dict={self.S: batch_s, self.a: batch_a, self.R: batch_r, self.S_: batch_s_})
        
    def store_transition(self, tempMemory):
        # replace the old memory with new memory
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = tempMemory
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        init_w = tf.random_normal_initializer(0., 0.01)
        init_b = tf.constant_initializer(0.01)
        hidden1 = 64
        hidden2 = 32
#        hiddeb3 = 64
        with tf.variable_scope(scope):
            layer1 = tf.layers.dense(s, hidden1, activation=tf.nn.relu, name='l1', trainable=trainable)
            layer2 = tf.layers.dense(layer1, hidden2, activation=tf.nn.relu, name='l2', trainable=trainable)
#            layer3 = tf.layers.dense(layer2, hidden3, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.layers.dense(layer2, self.a_dim, activation=tf.nn.sigmoid, name='a', kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
            return a

    def _build_c(self, s, a, scope, trainable):
        init_w = tf.random_normal_initializer(0., 0.01)
        init_b = tf.constant_initializer(0.01)
        with tf.variable_scope(scope):
#            hidden1 = 32
            hidden2 = 64
            hidden3 = 32
            layer_s = tf.layers.dense(s, hidden2, activation=tf.nn.relu, name='l1_s', trainable=trainable)
#            layer_s = tf.layers.dense(layer_s, hidden2, activation=None, name='l2_s', trainable=trainable)
            layer_a = tf.layers.dense(a, hidden2, activation=None, name='a', trainable=trainable)
            net = tf.layers.dense(tf.keras.layers.concatenate([layer_s, layer_a]), hidden3, activation=tf.nn.relu, name='cat', kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
            net = tf.layers.dense(net, self.a_dim, activation=None, name='Q', kernel_initializer=init_w, trainable=trainable)
            return net
  
    def save(self, DIR, episode):
        if DIR is None:
            DIR = "./temp/"
        modelName = datetime.datetime.now().strftime("%y_%m_%d") + ".ckpt"
        self.saver.save(self.sess, DIR+modelName, global_step = episode)
        print("Model:{} is saved!".format(modelName))
        
    def load(self, DIR, load=False):
        if load:
            if len(os.listdir(DIR)) == 0:
                print("There is not any models in the {}! Restart training...\n".format(DIR))
                time.sleep(1)
                return False
            else:
                print("Loading succeed!")
                self.saver.restore(self.sess, tf.train.latest_checkpoint(DIR))
#                self.newsaver = tf.train.import_meta_graph('saveModel/19_03_12.ckpt-1050.meta')
#                self.newsaver.restore(self.sess, 'saveModel/19_03_12.ckpt-1050')
                return True
        else:
            print("Start training...\n")
            return False
    
    def close(self):
        self.sess.close()
        
class OUNoise(object):
    def __init__(self, action_space=None, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=10000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = np.asarray(action_space, dtype=np.float32).shape[0]
        self.low          = np.zeros((self.action_dim), dtype=np.float32) 
        self.high         = np.ones((self.action_dim), dtype=np.float32)
        self.reset()
        # set the numpy random seed
        np.random.seed(2)
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def exploration(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class Discretization(object):
    def __init__(self, k=[2, 2], action_space=None, method='euclidean'):
        self.k1 = k[0]
        self.k2 = k[1]
        self.lightLow   = action_space[0][0]
        self.lightHigh  = action_space[0][1]
        self.light_step = action_space[0][2]
        self.expLow     = action_space[1][0]
        self.expHigh    = action_space[1][1]
        self.exp_step   = action_space[1][2]
        self.action_dim = np.asarray(action_space, dtype=np.float32).shape[0]
        # create discrete space of light (0-255) and then normalize the space
        lightDS = (np.arange(action_space[0][0], action_space[0][1]+self.light_step, self.light_step, dtype=np.int16)-self.lightLow) / (self.lightHigh-self.lightLow)
        # reshape the space to the (255, 1)
        lightDS = lightDS.reshape(lightDS.shape[0], 1)
        # create discrete space of exposureTime(9-19) and then normalize the space
        ExpDS = (np.arange(action_space[1][0], action_space[1][1]+self.exp_step, self.exp_step, dtype=np.float64)[:-1]-self.expLow) / (self.expHigh-self.expLow)
        # reshape like the before
        ExpDS = ExpDS.reshape(ExpDS.shape[0], 1)
        # creat the KDtree
        self.treeLight = KDTree(lightDS, leaf_size=lightDS.shape[0], metric=method)
        self.treeExp   = KDTree(ExpDS, leaf_size=ExpDS.shape[0], metric=method)
        
    def action(self, action, state):
        # map to the exposure time and lihgt space
        k_light1 = (self.treeLight.query(np.array([[action[0]]]), k=self.k1, return_distance=False)*self.light_step+self.lightLow).T
        k_exp1   = (self.treeExp.query(np.array([[action[1]]]), k=self.k2, return_distance=False)*self.exp_step+self.expLow).T
        k_light2 = (self.treeLight.query(np.array([[action[2]]]), k=self.k1, return_distance=False)*self.light_step+self.lightLow).T
        k_exp2   = (self.treeExp.query(np.array([[action[3]]]), k=self.k2, return_distance=False)*self.exp_step+self.expLow).T
        k_light3 = (self.treeLight.query(np.array([[action[4]]]), k=self.k1, return_distance=False)*self.light_step+self.lightLow).T
        k_exp3   = (self.treeExp.query(np.array([[action[5]]]), k=self.k2, return_distance=False)*self.exp_step+self.expLow).T
        
        # calculate the all of state and the action by fast method
        A = np.array(np.meshgrid(k_light1, k_exp1, k_light2, k_exp2, k_light3, k_exp3), dtype=np.float32).T.reshape(-1, self.action_dim)
        S = np.tile(state, (self.k1**3*self.k2**3, 1))
        
        return S, A
    
"""
Test Code:
Formate of action_space 
KDtree test
  
action_space = [[0, 255, 1], [9, 19, 0.1]]
discrete_act = Discretization(k = [3, 3], action_space=action_space)
action = np.array([[0.1, 0.6, 0.5, 0.4, 0.2, 0.7]])
state = np.array([[1.5, 2.3, 4.7]])
s,a = discrete_act.action(action, state)
print("finish")

"""