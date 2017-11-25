# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:01:49 2017

@author: rinziii
"""

from os import chdir
chdir('C:\\Users\\rinziii\\Documents\\codes_that_works')


import pandas as pd
import numpy as np

# importing the dataset
dataset = pd.read_csv('churn\\bank.csv')

X_ = dataset.iloc[:, 3:13].values
y_ = dataset.iloc[:, 13].values

#Encoding the X data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X_[:, 1] = labelencoder_X_1.fit_transform(X_[:, 1])

labelencoder_X_2 = LabelEncoder()
X_[:, 2] = labelencoder_X_2.fit_transform(X_[:, 2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X_ = onehotencoder.fit_transform(X_).toarray()
X_ = X_[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size = 0.2, random_state=41)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


X_train = np.float32(X_train)
y_train = np.float32(y_train)



import tensorflow as tf

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, X_train.shape[1]])
y = tf.placeholder(tf.float32,)     


def build_NN(data):
    
    l1_node = 6
    l2_node = 6
    n_class = 1
        
    # layer architecture
    layer_1 = {'weights': tf.get_variable(name ='w1', shape=[11, l1_node], initializer = tf.contrib.layers.xavier_initializer()),
               'bias': tf.zeros([l1_node])}

    
    layer_2 = {'weights': tf.get_variable(name='w2', shape=[l1_node, l2_node], initializer = tf.contrib.layers.xavier_initializer()),
               'bias': tf.zeros([l2_node])}
    
    

    output_layer = {'weights': tf.get_variable(name='w4', shape=[l2_node, n_class], initializer = tf.contrib.layers.xavier_initializer()),
               'bias': tf.zeros([n_class])}
    
    # model
    Z1 = tf.matmul(data, layer_1['weights']) + layer_1['bias']
    A1 = tf.nn.relu(Z1)
    
    Z2 = tf.matmul(A1, layer_2['weights']) + layer_2['bias']
    A2 = tf.nn.relu(Z2)

    output = tf.matmul(A2, output_layer['weights']) + output_layer['bias']
    
    return output


def train_model(X_train, y_train):
    y_hat = build_NN(X_train)
    y_train = y_train.reshape((y_train.shape[0], 1))
    
    y_hat = tf.cast(y_hat, tf.float32)
    
    # define cost function 
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=y_train))
    
    # optimizer
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    num_epoch = 2000
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(num_epoch):
            _, c = sess.run([optimizer, cost], feed_dict={X:X_train, y:y_train})
            
            print('Epoch', epoch, 'completed out of', num_epoch, 'loss', c)
        
        y_hat = sess.run(y_hat > 0.5)
        y_hat = tf.cast(y_hat, tf.float32)
        correct = tf.equal(y_hat, y)
        
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        
        print("Accuracy:", accuracy.eval({X:X_train, y:y_train})*100, '%')

train_model(X_train, y_train)
            
    


    