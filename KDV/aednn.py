#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import scipy.io
from pyDOE import lhs
import math


# In[2]:


RandomSeed = 1235
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
        self.b = tf.Variable(value_float64, dtype=tf.float64)
        
        value_float64=  1.0/(math.cosh(5*0.5)-1)       
        
        self.a = tf.constant(value_float64, dtype=tf.float64)   
     
        
    
        
        self.weights_values = weights_values        
        self.biases_values = biases_values    

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))    

        self.x_f_tf = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float64, shape=[None, self.t.shape[1]])
        
        self.x_lb_tf = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.t_b_tf = tf.placeholder(tf.float64, shape=[None, self.t.shape[1]])
        self.x_ub_tf = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        
        self.x_tf = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float64, shape=[None, self.t.shape[1]])
        
        self.u_tf = tf.placeholder(tf.float64, shape=[None, self.u.shape[1]])

        self.u_pred, _ ,self.uuuuuu = self.net_AC(self.x_tf, self.t_tf)
        self.u_lb_pred, self.ux_lb_pred,_ = self.net_AC(self.x_lb_tf, self.t_b_tf)
        self.u_ub_pred, self.ux_ub_pred,_ = self.net_AC(self.x_ub_tf, self.t_b_tf)

        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)
        


        
        self.lossS = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))                                              
        self.lossB = tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred)) + tf.reduce_mean(tf.square(self.ux_lb_pred - self.ux_ub_pred))
        self.lossfu =  tf.reduce_mean(tf.square(self.f_pred))    
        

        self.optimizer_Adam = tf.train.AdamOptimizer()

        self.loss  =   self.lossS + 20 *   self.lossB + self.lossfu
             
        
        
        
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
            W = tf.Variable(tf.zeros([layers[l], layers[l+1]], dtype=tf.float64), dtype=tf.float64)
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float64), dtype=tf.float64)
            weights.append(W)
            biases.append(b)        
        return weights, biases
    
    
    def initialize_NN2(self, layers):        
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


    def custom_function(self,t):
        
        ab= self.b     
        
        condition2 = tf.less_equal(t, 0.5)
        condition3 = tf.less_equal(t, ab)
        condition4 = tf.less_equal(ab, 1)
        
        BB=2*(t-0.5)
        
        
        B=1/(ab-0.5)*(t-0.5)
        
        return tf.where(condition2, tf.zeros_like(t), tf.where(condition4,tf.where(condition3, -2*B**3+3*B**2,tf.zeros_like(t)+1),-2*BB**3+3*BB**2))
    
    def neural_net(self, x,t, weights, biases,weights_values,biases_values):

        tt=2*t
        x = (x+1)/2
        num_layers = len(weights) + 1
        H = tf.concat([tt,x],1)
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            W1 = weights_values[l]
            b1 = biases_values[l]   
            H0 = tf.add(tf.matmul(H, W), b)
            H1 = tf.add(tf.matmul(H, W1), b1)
            
            B = self.custom_function(t)   
            H0 = H0*B              
            
            H  = tf.add(H0,H1)
            H =  tf.tanh(H)
        
        W = weights[-1]
        b = biases[-1]
        W1 = weights_values[-1]
        b1 = biases_values[-1]  
        H0 = tf.add(tf.matmul(H, W), b)
        
        B = self.custom_function(t)     
        H0 = H0*B            
        
        H1 = tf.add(tf.matmul(H, W1), b1) 
        H  = tf.add(H0,H1)      
        Y = H
        return Y
    
    def net_AC(self, x, t):
        u = self.neural_net(x,t, self.weights, self.biases,self.weights_values, self.biases_values)
        u_x = tf.gradients(u, x)[0]
        u_t = tf.gradients(u, t)[0]

        return u, u_x, u_t
    
       
    

    def net_f(self, x, t):
        u, u_x, u_t = self.net_AC(x, t)
        u_xx = tf.gradients(u_x, x)[0]
        u_xxx = tf.gradients(u_xx, x)[0]            
        f_u = u_t +  u*u_x + 0.0025*u_xxx
        return f_u
    
    def callback(self, loss, a,b,lossfu, lossS, lossB):
        sss=self.hsadasjd
        if sss%1000==0:
            print('Loss: %.6e, Lossfu: %.3e, LossS: %.3e, LossB: %.3e ' % (loss, lossfu, lossS, lossB))
        sss=sss+1
        self.hsadasjd=sss 
        self.hh.append(a)      
        self.hh1.append(b)        
    def train(self, nIter, Nf, Nn, Nb):

        X_train = self.lbp + (self.ubp-self.lbp)*lhs(2, Nf)
        self.xtrain_f = X_train[:,0:1]
        self.ttrain_f = X_train[:,1:2] 
        
        X_lb_train = self.lbp + [0,self.ubp[1]-self.lbp[1]]*lhs(2, Nb)
        self.xtrain_lb = X_lb_train[:,0:1]
        self.ttrain_b = X_lb_train[:,1:2]
        
        X_ub_train = [self.ubp[0],self.lbp[1]] + [0,self.ubp[1]-self.lbp[1]]*lhs(2, Nb)
        self.xtrain_ub = X_ub_train[:,0:1]
        
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u,
                   self.x_lb_tf: self.xtrain_lb, self.t_b_tf: self.ttrain_b, 
                   self.x_ub_tf: self.xtrain_ub, 
                   self.x_f_tf: self.xtrain_f, self.t_f_tf: self.ttrain_f}

        for it in range(nIter):
            loss_value = self.sess.run(self.loss, tf_dict)
            lossfu = self.sess.run(self.lossfu, tf_dict)
            lossS = self.sess.run(self.lossS, tf_dict)
            lossB = self.sess.run(self.lossB, tf_dict)
            
            self.sess.run(self.train_op_Adam, tf_dict)
                
            ab= self.sess.run(self.a)
            self.hh.append(ab)       
            abc= self.sess.run(self.b)
            self.hh1.append(abc)    

            print(abc)                 
            if it % 1000 == 0:
                print('It: %d, Loss: %.6e, Lossfu: %.3e, LossS: %.3e, LossB: %.3e' % (it, loss_value, lossfu, lossS, lossB))
                

        
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, method = 'L-BFGS-B', options = {'maxiter': 20000,'maxfun': 20000,'maxcor': 50,'maxls': 50,'ftol' : 1.0 * np.finfo(float).eps})                                                                                                         
        self.optimizer.minimize(self.sess, feed_dict = tf_dict, fetches = [self.loss,self.a,self.b,  self.lossfu, self.lossS, self.lossB], loss_callback = self.callback)        
                                    
    
    def predict(self, x, t):
        
        tf_dict = {self.x_tf: x, self.t_tf: t}
        u_star = self.sess.run(self.u_pred, tf_dict)
        u_starssss = self.sess.run(self.uuuuuu, tf_dict)        
        return u_star,u_starssss

    def saver(self, string):
        self.save.save(self.sess, 'ckpt/'+string)
        
        
    def sssss(self):
        return self.hh,  self.hh1           
        
        
    def restore(self):
        model_file = tf.train.latest_checkpoint('ckpt/')
        self.save.restore(self.sess, model_file)


# In[4]:


import pickle


# In[5]:


with open('bcweights.pkl', 'rb') as f:
    weights_values = pickle.load(f)


# In[6]:


with open('bcweights1.pkl', 'rb') as f:
    biases_values = pickle.load(f)


# In[7]:


if __name__ == "__main__": 
           
    
    # Doman bounds
    lb = np.array([-1, 0])
    ub = np.array([1, 0.5])
    
    lb1 = np.array([-1, 0.5])
    ub1 = np.array([1, 1.0]) 
    
    layers = [2,30,30,30,1]
    
    data = scipy.io.loadmat('kdv.mat')
    
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = data['usol']
    
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
      
    model2 = PtPINNsss(X0, T0, U0, lb, ub,lb1, ub1, layers,weights_values,biases_values)                        
    model2.train(5000 ,4000, 400,400)  
    model2.saver('testmodel.ckpt')
    


    u_pred ,_= model2.predict(X_star[:,0:1],X_star[:,1:2])

    erroru = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    erroru1 = np.linalg.norm(u_star-u_pred,1)/len(X_star)
    erroruinf = np.linalg.norm(u_star-u_pred,np.inf)
    
    print('randorm seed: %d' % (RandomSeed))

    print('Error2 u: %e' % (erroru))
    print('Error1 u: %e' % (erroru1))
    print('Errorf u: %e' % (erroruinf))


# In[9]:


data = scipy.io.loadmat('kdv.mat')
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = data['usol']

X, T = np.meshgrid(x,t)
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.T.flatten()[:,None]
u_pred ,f_pred= model2.predict(X_star[:,0:1],X_star[:,1:2])

erroru = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
erroru1 = np.linalg.norm(u_star-u_pred,1)/len(X_star)
erroruinf = np.linalg.norm(u_star-u_pred,np.inf)

print('randorm seed: %d' % (RandomSeed))
print('Error2 u: %e' % (erroru))
print('Error1 u: %e' % (erroru1))
print('Errorf u: %e' % (erroruinf))
U_pred=u_pred.reshape(201,512).T
U_preds=U_pred


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




