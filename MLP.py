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
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# In[3]:


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
a = -1
b = 1
N = 2500
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


# In[4]:


def Normalizacja(X):
    mu = np.mean(X)
    sig = statistics.stdev(X)
    data = (X - mu) / sig;
    return data

X_train_std = Normalizacja(X)
X_test_std = Normalizacja(X_org)
Y_train = Y
Y_test = Y_org


# In[5]:


model = keras.Sequential()
model.add(layers.InputLayer(input_shape=(1,)))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(30, activation="relu"))
model.add(layers.Dense(10, activation="relu"))
model.add(layers.Dense(30, activation="relu"))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(1))

model.compile(loss='mse', optimizer='sgd')
model.summary()


# In[6]:


history = model.fit(X_train_std, Y_train, epochs=300, batch_size=128, verbose=0)
plt.plot(history.history['loss'])
plt.show()


# In[7]:


predictions = model.predict(X_test_std)
RMSE = math.sqrt(np.mean(predictions- Y_test)**2)
print(RMSE)

plt.plot(X_test_std, predictions)
plt.plot(X_test_std, Y_test)
plt.show()


# In[8]:


X_train_std_rnn = np.reshape(X_train_std, (X_train_std.shape[0], 1, 1))
X_test_std_rnn = np.reshape(X_test_std, (X_test_std.shape[0], 1, 1))


# In[9]:


model_rnn = keras.Sequential()
model_rnn.add(layers.LSTM(120, input_shape=(1, 1)))
model_rnn.add(layers.Dense(1))

model_rnn.compile(loss='mse', optimizer='adam')
model_rnn.summary()


# In[10]:


history_rnn = model_rnn.fit(X_train_std_rnn, Y_train, epochs=1000, batch_size=64, verbose=0)
plt.plot(history_rnn.history['loss'])
plt.show()


# In[11]:


predictions_rnn = model_rnn.predict(X_test_std_rnn)
RMSE_rnn = math.sqrt(np.mean(predictions_rnn - Y_test)**2)
print(RMSE_rnn)
plt.plot(X_test_std, predictions_rnn)
plt.plot(X_test_std, Y_test)
plt.show()

