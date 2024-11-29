#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import scipy.io
from pyDOE import lhs
import math


# In[2]:


RandomSeed = 1236
np.random.seed(RandomSeed)
tf.set_random_seed(RandomSeed)


# In[3]:


class PtPINN:
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

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        
        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_b_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.u_pred, _ ,_ = self.net_AC(self.x_tf, self.t_tf)
        self.u_lb_pred, self.ux_lb_pred,_ = self.net_AC(self.x_lb_tf, self.t_b_tf)
        self.u_ub_pred, self.ux_ub_pred,_ = self.net_AC(self.x_ub_tf, self.t_b_tf)

        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)
        
        
        self.lossS = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
                                              
        self.lossB = tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred)) + tf.reduce_mean(tf.square(self.ux_lb_pred - self.ux_ub_pred))
                                             
        self.lossfu = tf.reduce_mean(tf.square(self.f_pred))        
        

        self.optimizer_Adam = tf.train.AdamOptimizer()

        
        self.loss  =  64 * self.lossS  + self.lossfu
    
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
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32))
    
    
    
    def input_encoding(self, t, x):    
        # 创建频率数组
        pi_terms = tf.range(1, 11, dtype=tf.float32) * math.pi

        # 计算cos和sin值
        cos_vals = tf.cos(pi_terms * x)
        sin_vals = tf.sin(pi_terms * x)

        # 拼接成一个向量
        H = tf.concat([t, 1 + 0 * tf.cos(math.pi * x), cos_vals, sin_vals], axis=1)

        return H

        
        
    def neural_net(self, x,t, weights, biases):
        num_layers = len(weights) + 1
        t = 2*t 
        H = self.input_encoding(t, x)
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
        
        f_u = u_t - 0.0001*u_xx+5*u**3-5*u
        
        return f_u
    
        
    def train(self, nIter, Nf, Nb):

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
        # L-BFGS optimizer    
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, method = 'L-BFGS-B', options = {'maxiter': 50000,
                                                                           'maxfun': 50000, 'maxcor': 50, 'maxls': 50, 'ftol' : 1.0 * np.finfo(float).eps})                                                                                                         
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
           
    lb = np.array([-1.0, 0])
    ub = np.array([1.0, 1])


    ubp = np.array([1.0,0.5])

    
    layers = [22, 50, 50, 50, 50, 1]
    
    data = scipy.io.loadmat('C:\\Users\\User\\Desktop\\data\\AC.mat')
    
    t = data['tt'].flatten()[:,None][0:101] 
    x = data['x'].flatten()[:,None]
    Exact = data['uu'][:,0:101]
    

    
    X, T = np.meshgrid(x,t)
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.T.flatten()[:,None]

    def IC(x):
        u = x**2*np.cos(np.pi*x)
        return u

    N0 = 200
    x=np.linspace(-1,1,N0).flatten()[:,None]  
    X0 =x
    T0 = np.full((N0,1), lb[1])
    U0 = IC(X0)
      
    model = PtPINN(X0, T0, U0, lb, ub, ubp, layers)                          
 


# In[5]:


model.train(5000, 5000, 200)    


# In[6]:


u_pred = model.predict(X_star[:,0:1],X_star[:,1:2])

erroru = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
erroru1 = np.linalg.norm(u_star-u_pred,1)/len(X_star)
erroruinf = np.linalg.norm(u_star-u_pred,np.inf)

print('randorm seed: %d' % (RandomSeed))
print('Error2 u: %e' % (erroru))
print('Error1 u: %e' % (erroru1))
print('Errorf u: %e' % (erroruinf))


# In[7]:


weights_values = model.sess.run(model.weights)
biases_values = model.sess.run(model.biases)


# In[8]:


import pickle
with open('bcweights.pkl', 'wb') as f:
    pickle.dump(weights_values, f)
with open('bcweights1.pkl', 'wb') as f:
    pickle.dump(biases_values, f)


# In[ ]:





# In[ ]:




