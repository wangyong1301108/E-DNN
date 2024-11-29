#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import scipy.io
from pyDOE import lhs
import math


# In[2]:


RandomSeed = 9
np.random.seed(RandomSeed)
tf.set_random_seed(RandomSeed)


# In[3]:


class PtPINNsss:
    # Initialize the class
    def __init__(self, x, t, u, lb, ub, lbp,ubp, layers,weights_values,biases_values):
        
        X = np.concatenate([x, t], 1)

        self.X = X
        
        self.x = X[:,0:1]
        self.t = X[:,1:2]
        
        self.u = u      
        self.lb = lb
        self.ub = ub
        self.hsadasjd=0
        self.ubp = ubp
        self.lbp = lbp        
        # Initialize NNs
        self.layers = layers
        self.hh=[]
        self.hh1=[]      
        
        self.weights, self.biases = self.initialize_NN(layers)
        

        value_float64 = 3.0/4
        self.b = tf.Variable(value_float64, dtype=tf.float32)
        
        self.weights_values = weights_values        
        self.biases_values = biases_values    

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))    

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        
        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_b_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.u_pred, _ ,self.uuuuuu = self.net_AC(self.x_tf, self.t_tf)
        self.u_lb_pred, self.ux_lb_pred,_ = self.net_AC(self.x_lb_tf, self.t_b_tf)
        self.u_ub_pred, self.ux_ub_pred,_ = self.net_AC(self.x_ub_tf, self.t_b_tf)

        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)

        self.lossfu =  tf.reduce_mean(tf.square(self.f_pred))    
        

        self.optimizer_Adam = tf.train.AdamOptimizer()

        self.loss  =  self.lossfu
        
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
            W = tf.Variable(tf.zeros([layers[l], layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
    
    
    def custom_function(self,t):
        
        ab= self.b     
        
        condition2 = tf.less_equal(t, 0.5)
        condition3 = tf.less_equal(t, ab)
        condition4 = tf.less_equal(ab, 1)
        
        BB=2*(t-0.5)
        
        
        B=1/(ab-0.5)*(t-0.5)
        
        return tf.where(condition2, tf.zeros_like(t), tf.where(condition4,tf.where(condition3, -2*B**3+3*B**2,tf.zeros_like(t)+1),-2*BB**3+3*BB**2))
    
 

    def input_encoding(self, t, x):    
        # 创建频率数组
        pi_terms = tf.range(1, 11, dtype=tf.float32) * math.pi

        # 计算cos和sin值
        cos_vals = tf.cos(pi_terms * x)
        sin_vals = tf.sin(pi_terms * x)

        # 拼接成一个向量
        H = tf.concat([t, 1 + 0 * tf.cos(math.pi * x), cos_vals, sin_vals], axis=1)

        return H


    def neural_net(self, x, t, weights, biases, weights_values, biases_values):
        X = tf.concat([x, t], 1)
        tt = 2 * t

        # 提前计算 t 的自定义函数值，避免重复调用
        B = self.custom_function(t)

        # 输入编码
        H = self.input_encoding(tt, x)
        num_layers = len(weights) + 1

        # 前向传播层
        for l in range(num_layers - 2):
            H0 = tf.add(tf.matmul(H, weights[l]), biases[l]) * B
            H1 = tf.add(tf.matmul(H, weights_values[l]), biases_values[l])
            H = tf.tanh(H0 + H1)

        # 最后一层
        H0 = tf.add(tf.matmul(H, weights[-1]), biases[-1]) * B
        H1 = tf.add(tf.matmul(H, weights_values[-1]), biases_values[-1])
        Y = H0 + H1

        return Y

    
    def net_AC(self, x, t):
        u = self.neural_net(x,t, self.weights, self.biases,self.weights_values, self.biases_values)
        u_x = tf.gradients(u, x)[0]
        u_t = tf.gradients(u, t)[0]

        return u, u_x, u_t
          
    

    def net_f(self, x, t):
        u, u_x, u_t = self.net_AC(x, t)
        
        u_xx = tf.gradients(u_x, x)[0]
        
        f_u = u_t - 0.0001*u_xx+5*u**3-5*u
        
        return f_u
    
    def callback(self, loss,b,lossfu, lossS, lossB):
        sss=self.hsadasjd
        if sss%1000==0:
            print('Loss: %.6e, Lossfu: %.3e, LossS: %.3e, LossB: %.3e ' % (loss, lossfu, lossS, lossB))
        sss=sss+1
        self.hsadasjd=sss      
        self.hh1.append(b)   
        
    def train(self, nIter, Nf, Nb):

        X_train = self.lbp + (self.ubp-self.lbp)*lhs(2, Nf)
        self.xtrain_f = X_train[:,0:1]
        self.ttrain_f = X_train[:,1:2] 
        
        X_lb_train = self.lbp + [0,self.ubp[1]-self.lbp[1]]*lhs(2, Nb)
        self.xtrain_lb = X_lb_train[:,0:1]
        self.ttrain_b = X_lb_train[:,1:2]
        
        self.xtrain_ub = -1*X_lb_train[:,0:1]
        
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u,
                   self.x_lb_tf: self.xtrain_lb, self.t_b_tf: self.ttrain_b, 
                   self.x_ub_tf: self.xtrain_ub, 
                   self.x_f_tf: self.xtrain_f, self.t_f_tf: self.ttrain_f}

        for it in range(nIter):

            abc= self.sess.run(self.b)
            self.hh1.append(abc)    
            self.sess.run(self.train_op_Adam, tf_dict)
                        
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, method = 'L-BFGS-B', options = {'maxiter': 50000,'maxfun': 50000,'maxcor': 50,'maxls': 50,'ftol' : 1.0 * np.finfo(float).eps})                                                                                                         
        self.optimizer.minimize(self.sess, feed_dict = tf_dict)        
                                    
    
    def predict(self, x, t):
        
        tf_dict = {self.x_tf: x, self.t_tf: t}
        u_star = self.sess.run(self.u_pred, tf_dict)
        u_starssss = self.sess.run(self.uuuuuu, tf_dict)        
        return u_star,u_starssss

    def saver(self, string):
        self.save.save(self.sess, 'ckpt/'+string)
        
        
    def sssss(self):
        return self.hh         
        
        
    def restore(self):
        model_file = tf.train.latest_checkpoint('ckpt/')
        self.save.restore(self.sess, model_file)


# In[4]:


import pickle
with open('bcweights.pkl', 'rb') as f:
    weights_values = pickle.load(f)
with open('bcweights1.pkl', 'rb') as f:
    biases_values = pickle.load(f)


# In[5]:


if __name__ == "__main__": 
           
    
    # Doman bounds
    lb = np.array([-1.0, 0])
    ub = np.array([1.0, 0.5])


    lbp = np.array([-1.0, 0.5])    
    ubp = np.array([1.0,1.0])

    
    layers = [22, 50, 50, 50, 50, 1]
    
    data = scipy.io.loadmat('C:\\Users\\User\\Desktop\\data\\AC.mat')
    
    t = data['tt'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = data['uu']
    
    X, T = np.meshgrid(x,t)
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.T.flatten()[:,None]

    def IC(x):
        u = x**2*np.cos(np.pi*x)
        return u

    N0 = 2
    x=np.linspace(-1,1,N0).flatten()[:,None]  
    X0 =x
    T0 = np.full((N0,1), lb[1])
    U0 = IC(X0)
      
    model2 = PtPINNsss(X0, T0, U0, lb, ub,lbp, ubp, layers,weights_values,biases_values)                        
    model2.train(5000, 5000, 200)  
    model2.saver('testmodel.ckpt')
    


    u_pred ,_= model2.predict(X_star[:,0:1],X_star[:,1:2])

    erroru = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    erroru1 = np.linalg.norm(u_star-u_pred,1)/len(X_star)
    erroruinf = np.linalg.norm(u_star-u_pred,np.inf)
    
    print('randorm seed: %d' % (RandomSeed))
    print('Error2 u: %e' % (erroru))
    print('Error1 u: %e' % (erroru1))
    print('Errorf u: %e' % (erroruinf))


# In[ ]:




