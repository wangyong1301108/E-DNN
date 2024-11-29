#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, '../../Utilities/')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import math


# In[2]:


import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
RandomSeed = 1236
np.random.seed(RandomSeed)
tf.set_random_seed(RandomSeed)


# In[3]:


class PtPINN:
    def __init__(self, x, t, u, lb, ub,layers):
        
        X = np.concatenate([x, t], 1)

        self.X = X
        
        self.x = X[:,0:1]
        self.t = X[:,1:2]
        
        self.u = u
        self.hsadasjd=0 
        self.lb = lb
        self.ub = ub
        
        
        # Initialization
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers) 

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))    

        self.x_f_tf = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float64, shape=[None, self.t.shape[1]])
        
        self.x_lb_tf = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.t_b_tf = tf.placeholder(tf.float64, shape=[None, self.t.shape[1]])
        self.x_ub_tf = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.x_tf = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float64, shape=[None, self.t.shape[1]])
        self.u_tf = tf.placeholder(tf.float64, shape=[None, self.u.shape[1]])

        self.u_pred, _, _ = self.net_CE(self.x_tf, self.t_tf)
        self.u_lb_pred, _, _ = self.net_CE(self.x_lb_tf, self.t_b_tf)
        self.u_ub_pred, _, _ = self.net_CE(self.x_ub_tf, self.t_b_tf)

        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)
     

        # Loss function(MSE)
        
        self.lossS = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
                                              
        self.lossB = tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred))
                                             
        self.lossfu = tf.reduce_mean(tf.square(self.f_pred))    
        
    
        self.optimizer_Adam = tf.train.AdamOptimizer()

        
        self.loss  =  self.lossS + 20 * self.lossB + self.lossfu
        
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)      
        

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.save = tf.train.Saver(max_to_keep=1)
              
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float64), dtype=tf.float64)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64))
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_CE(self, x, t):
        X = tf.concat([x,t],1)
        u = self.neural_net(X, self.weights, self.biases)
        u_x = tf.gradients(u, x)[0]
        u_t = tf.gradients(u, t)[0]
        return u, u_x, u_t

    
    
    def net_f(self, x, t):
        u, u_x, u_t = self.net_CE(x, t)
        f_u = u_t +40*u_x 
        return f_u
    
 
        
    def train(self, nIter, Nf, Nb):
        # Resampling strategy
        X_train = self.lb + (self.ub-self.lb)*lhs(2, Nf)
        self.xtrain_f = X_train[:,0:1]
        self.ttrain_f = X_train[:,1:2] 
        
        X_lb_train = self.lb + [0,self.ub[1]-self.lb[1]]*lhs(2, Nb)
        self.xtrain_lb = X_lb_train[:,0:1]
        self.ttrain_b = X_lb_train[:,1:2]
        
        X_ub_train = [self.ub[0],0] + [0,self.ub[1]-self.lb[1]]*lhs(2, Nb)
        self.xtrain_ub = X_ub_train[:,0:1]
        
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u,
                   self.x_lb_tf: self.xtrain_lb, self.t_b_tf: self.ttrain_b, 
                   self.x_ub_tf: self.xtrain_ub, 
                   self.x_f_tf: self.xtrain_f, self.t_f_tf: self.ttrain_f}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, method = 'L-BFGS-B', options = {'maxiter': 50000,'maxfun': 50000,'maxcor': 50,'maxls': 50,'ftol' : 1.0 * np.finfo(float).eps})                                                                                                         
        self.optimizer.minimize(self.sess, feed_dict = tf_dict)        
                                    
    
    def predict(self, x, t):
        
        tf_dict = {self.x_tf: x, self.t_tf: t}
        u_star = self.sess.run(self.u_pred, tf_dict)
        
        return u_star

    def saver(self, string):
        self.save.save(self.sess, 'ckpt/'+string)

    def restore(self):
        model_file = tf.train.latest_checkpoint('ckpt/')
        self.save.restore(self.sess, model_file)


# In[4]:


if __name__ == "__main__": 
    # Doman bounds
    lb = np.array([0, 0])
    ub = np.array([2*np.pi, 0.5]) 
    
    layers = [2,100,100,100,100,1]
    
    def Exact_Solution(x, t):
        u = np.sin(x-40*t)
        return u

    N0 = 200
    x=np.linspace(0,2*np.pi,N0).flatten()[:,None]  
    X0 =x
    T0 = np.full((N0,1), lb[1])
    u0 = Exact_Solution(X0,T0)


# In[5]:


model = PtPINN(X0, T0, u0, lb, ub, layers)                       
model.train(5000, 5000, 200)


# In[16]:


x=np.linspace(0,2*np.pi,1200).flatten()[:,None]   
t=np.linspace(0,0.5,1200).flatten()[:,None]  
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_pred1=model.predict(X_star[:,0:1],X_star[:,1:2])
u_star1 = Exact_Solution(X_star[:,0:1],X_star[:,1:2])
u_star=u_star1.flatten()[:,None]  
u_pred=u_pred1.flatten()[:,None] 
error_u1 = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
error_u2 = np.linalg.norm(u_star-u_pred,1)/len(u_star)
error_u3 = np.linalg.norm(u_star-u_pred,np.inf)
print('二范数Error u: %e' % (error_u1))
print('平均绝对Error u: %e' % (error_u2))
print('无穷范数Error u: %e' % (error_u3))


# In[20]:


weights_values = model.sess.run(model.weights)
biases_values = model.sess.run(model.biases)
import pickle
with open('weights.pkl', 'wb') as f:
    pickle.dump(weights_values, f)
with open('weights1.pkl', 'wb') as f:
    pickle.dump(biases_values, f)


# In[ ]:





# In[ ]:





# In[ ]:




