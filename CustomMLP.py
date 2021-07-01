#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import statistics

from tensorflow import keras
import keras.backend as K
from tensorflow.keras import layers

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops


# In[2]:


def Nieliniowosc(X):
    V = 3 * np.power(X, 3)
    return V

def System_Hammersteina(X, Z, gamma):
    V = Nieliniowosc(X)
    V_pom = np.concatenate((np.zeros(len(gamma)-1), V))
    Y = np.zeros(len(X))
    for i in range(len(X)):
        Y[i] = sum(gamma * np.flip(V_pom[i:i+len(gamma)]))
    Y = Y + Z
    return Y

def Odp_impulsowa(gamma):
    for i in range(len(gamma)):
        gamma[i] = gamma[i]**i
    return gamma

# Dane pomiarowe na określonym przedziale
a = -3
b = 3
N = 12000
X = a + (b - a) * np.random.rand(N)
Z = np.random.randn(len(X))
pom = 0.9 * np.ones(25)
gamma = Odp_impulsowa(pom)
Y = System_Hammersteina(X, Z, gamma)

# Oryginalna charakterystyka systemu
X_org = np.linspace(a, b, len(X))
Y_org = Nieliniowosc(X_org)

plt.plot(X, Y,'.')
plt.plot(X_org, Y_org)
plt.show()
plt.scatter(np.linspace(1,len(gamma),len(gamma)), gamma)
plt.show()


# In[3]:


def Normalizacja(X):
    mu = np.mean(X)
    sig = statistics.stdev(X)
    data = (X - mu) / sig;
    return data

X_train_std = Normalizacja(X)
X_test_std = Normalizacja(X_org)
Y_train = Y
Y_test = Y_org


# In[4]:


class customSGD(keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, name="customSGD", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "pv") 
    
    @tf.function
    def _resource_apply_dense(self, grad, var):
        lr_t = self.lr
        new_var = var - grad * lr_t
        pv_var = self.get_slot(var, "pv")
        pv_var.assign(var)
        var.assign(new_var)

class customSGDmom(keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, gamma=0.9, name="customSGDmom", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self.gamma = gamma
    
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "pv")
        for var in var_list:
            self.add_slot(var, "vt")
    
    @tf.function
    def _resource_apply_dense(self, grad, var):
        lr_t = self.lr
        gamma_t = self.gamma
        p_vt = self.get_slot(var, "vt")
        vt = gamma_t * p_vt + grad * lr_t
        p_vt.assign(vt)
        new_var = var - vt
        pv_var = self.get_slot(var, "pv")
        pv_var.assign(var)
        var.assign(new_var)

class customADAM(keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, name="customADAM", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self.b1 = beta1
        self.b2 = beta2
        
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "pv")
        for var in var_list:
            self.add_slot(var, "mt")
        for var in var_list:
            self.add_slot(var, "vt")
        for var in var_list:
            self.add_slot(var, "t")
    
    @tf.function
    def _resource_apply_dense(self, grad, var):
        lr_t = self.lr
        b1_t = self.b1
        b2_t = self.b2
        
        p_mt = self.get_slot(var, "mt")
        p_vt = self.get_slot(var, "vt")
        p_t = self.get_slot(var, "t")
        
        t = p_t + 1
        p_t.assign(t)
        
        mt = b1_t * p_mt + (1 - b1_t) * grad
        vt = b2_t * p_vt + (1 - b2_t) * grad**2
        
        p_mt.assign(mt)
        p_vt.assign(vt)
        
        m_est = mt/(1 - b1_t**t)
        v_est = vt/(1 - b2_t**t)
        
        new_var = var - (lr_t/(K.sqrt(v_est) + 0.00000001)) * m_est
        
        pv_var = self.get_slot(var, "pv")
        pv_var.assign(var)
        var.assign(new_var)

def customLossMSE(true, predicted):
    mse = tf.reduce_mean(tf.square(tf.subtract(predicted, true)))
    return mse


# In[5]:


keras.backend.clear_session()

model = keras.Sequential()
model.add(layers.InputLayer(input_shape=(1,)))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(1))

#optimizer = customSGD(learning_rate=0.001)
#optimizer = customSGDmom(learning_rate=0.001, gamma=0.9)
optimizer = customADAM(learning_rate=0.001, beta1=0.9, beta2=0.999)

X_train_std_ten = np.reshape(X_train_std, (X_train_std.shape[0], 1, 1))
X_test_std_ten = np.reshape(X_test_std, (X_test_std.shape[0], 1, 1))

batch_size = 512
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_std_ten, Y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)


# In[6]:


epochs = 300
loss_history = []
suma = 0
num_iter = N/batch_size
for epoch in range(epochs):
    
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        
        with tf.GradientTape() as tape:
            
            logits = model(x_batch_train, training=True)
            y_batch_train = tf.dtypes.cast(y_batch_train, tf.float32)
            y_batch_train = tf.reshape(y_batch_train, (len(y_batch_train), 1))
            loss_value = customLossMSE(y_batch_train, logits)
            suma = suma + loss_value
            
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
    loss_history.append(suma/num_iter)
    if(epoch % 20 == 0):
        print("Mean of training losses at epoch %d: %.4f" % (epoch, suma/num_iter))
    suma = 0
    
plt.plot(loss_history)
plt.show()


# In[7]:


predictions = model.predict(X_test_std_ten)
MSE = np.mean((predictions - Y_test.reshape(len(Y_test), 1))**2)
print(
    "Created model MSE: %.4f"
    % (float(MSE))
)

plt.plot(X_org, predictions)
plt.plot(X_org, Y_test)
plt.show()


# In[8]:


def I(x):
    if abs(x)<=1:
        return 1
    elif abs(x)>1:
        return 0
    
def K1(x):
    return 0.5*I(x)

def est_jadr(x,X,Y,hN,K):
    return sum([Y[i]*K((X[i]-x)/hN) for i in range(len(X))])/sum([K((X[i]-x)/hN) for i in range(len(X))])

y_pred_kernel = [est_jadr(i,X,Y,0.1,K1) for i in X_org]
MSE_kernel = np.mean((y_pred_kernel - Y_org)**2)
print(
    "Created model MSE: %.4f"
    % (float(MSE_kernel))
)

plt.plot(X_org, y_pred_kernel)
plt.plot(X_org, Y_org)

