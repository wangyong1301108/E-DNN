#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import scipy.io
from pyDOE import lhs
import math


# In[2]:


RandomSeed = 1234
np.random.seed(RandomSeed)
tf.set_random_seed(RandomSeed)


# In[3]:


class PtPINN:
    # Initialize the class
    def __init__(self, x, t, u, lb, ub, ubp, layers):
        
        X = np.concatenate([x, t], 1)

        self.X = X
        
        self.x = X[:,0:1]
        self.t = X[:,1:2]
        
        self.u = u
        self.hsadasjd=0
        self.lb = lb
        self.ub = ub

        self.ubp = ubp

        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)


        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))    

        self.x_f_tf = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float64, shape=[None, self.t.shape[1]])
        
        self.x_lb_tf = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.t_b_tf = tf.placeholder(tf.float64, shape=[None, self.t.shape[1]])
        self.x_ub_tf = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        
        self.x_tf = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float64, shape=[None, self.t.shape[1]])
        
        self.u_tf = tf.placeholder(tf.float64, shape=[None, self.u.shape[1]])

        self.u_pred, _ ,_ = self.net_AC(self.x_tf, self.t_tf)
        self.u_lb_pred, self.ux_lb_pred,_ = self.net_AC(self.x_lb_tf, self.t_b_tf)
        self.u_ub_pred, self.ux_ub_pred,_ = self.net_AC(self.x_ub_tf, self.t_b_tf)

        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)
        
        
        self.lossS = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
                                              
        self.lossB = tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred)) + tf.reduce_mean(tf.square(self.ux_lb_pred - self.ux_ub_pred))
                                             
        self.lossfu = tf.reduce_mean(tf.square(self.f_pred))        
        
        
        
        self.optimizer_Adam = tf.train.AdamOptimizer()

        self.loss  =  100 * self.lossS + self.lossB + self.lossfu
                  
        
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
    

        
        
    def neural_net(self, x,t, weights, biases):
        num_layers = len(weights) + 1
        H=tf.concat([2*t ,(x+1)/2],1)
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_AC(self, x, t):
        
        u = self.neural_net(x,t, self.weights, self.biases)
        u_x = tf.gradients(u, x)[0]
        u_t = tf.gradients(u, t)[0]

        return u, u_x, u_t
    
    

    def net_f(self, x, t):
        u, u_x, u_t = self.net_AC(x, t)
        
        u_xx = tf.gradients(u_x, x)[0]
        u_xxx = tf.gradients(u_xx, x)[0]            
        a=5.0
        b=0.5
        c=0.005
        
        f_u = u_t +  u*u_x + 0.0025*u_xxx
        
        return f_u
    
    def callback(self, loss, lossfu, lossS, lossB):
        sss=self.hsadasjd
        if sss%1000==0:
            print('Loss: %.6e, Lossfu: %.3e, LossS: %.3e, LossB: %.3e ' % (loss, lossfu, lossS, lossB))
        sss=sss+1
        self.hsadasjd=sss 
        
    def train(self, nIter, Nf, Nn, Nb):

        X_train = self.lb + (self.ubp-self.lb)*lhs(2, Nf)
        self.xtrain_f = X_train[:,0:1]
        self.ttrain_f = X_train[:,1:2] 
        
        X_lb_train = self.lb + [0,self.ubp[1]-self.lb[1]]*lhs(2, Nb)
        self.xtrain_lb = X_lb_train[:,0:1]
        self.ttrain_b = X_lb_train[:,1:2]
        
        self.xtrain_ub = -1*X_lb_train[:,0:1]
        
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u,
                   self.x_lb_tf: self.xtrain_lb, self.t_b_tf: self.ttrain_b, 
                   self.x_ub_tf: self.xtrain_ub, 
                   self.x_f_tf: self.xtrain_f, self.t_f_tf: self.ttrain_f}

        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 1000 == 0:
                
                loss_value = self.sess.run(self.loss, tf_dict)
                lossfu = self.sess.run(self.lossfu, tf_dict)
                lossS = self.sess.run(self.lossS, tf_dict)
                lossB = self.sess.run(self.lossB, tf_dict)

                print('It: %d, Loss: %.6e, Lossfu: %.3e, LossS: %.3e, LossB: %.3e' % (it, loss_value, lossfu, lossS, lossB))
        
        # L-BFGS optimizer    
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, method = 'L-BFGS-B', options = {'maxiter': 20000,
                                                                           'maxfun': 20000, 'maxcor': 50, 'maxls': 50, 'ftol' : 1.0 * np.finfo(float).eps})                                                                                                         
        self.optimizer.minimize(self.sess, feed_dict = tf_dict,fetches = [self.loss, self.lossfu, self.lossS, self.lossB],loss_callback = self.callback)        
                                    
    
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
           
    lb = np.array([-1.0, 0])
    ub = np.array([1.0, 1])

    # Pre-training interval
    ubp = np.array([1.0,0.5])
    
    layers = [2,30,30,30,1]
    
    data = scipy.io.loadmat('kdv.mat')
    
    t = data['t'].flatten()[:,None][0:101] 
    x = data['x'].flatten()[:,None]
    Exact = data['usol'][:,0:101]
    
    
    X, T = np.meshgrid(x,t)
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.T.flatten()[:,None]

    def IC(x):
        u = np.cos(np.pi*x)
        return u

    N0 = 400
    x=np.linspace(-1,1,N0).flatten()[:,None]  
    X0 =x
    T0 = np.full((N0,1), lb[1])
    U0 = IC(X0)
      
    model = PtPINN(X0, T0, U0, lb, ub, ubp, layers)                          

    model.train(5000,4000, 400, 400)    
    


    u_pred = model.predict(X_star[:,0:1],X_star[:,1:2])

    erroru = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    erroru1 = np.linalg.norm(u_star-u_pred,1)/len(X_star)
    erroruinf = np.linalg.norm(u_star-u_pred,np.inf)
    
    print('randorm seed: %d' % (RandomSeed))
    print('Training error in pre-training interval:(%.2f,%.2f)' % (lb[1], ubp[1]) ) 
    print('Error2 u: %e' % (erroru))
    print('Error1 u: %e' % (erroru1))
    print('Errorf u: %e' % (erroruinf))


# In[12]:


weights_values = model.sess.run(model.weights)
biases_values = model.sess.run(model.biases)


# In[13]:


import pickle


# In[14]:


with open('bcweights.pkl', 'wb') as f:
    pickle.dump(weights_values, f)


# In[15]:


with open('bcweights1.pkl', 'wb') as f:
    pickle.dump(biases_values, f)


# In[ ]:





# In[ ]:




