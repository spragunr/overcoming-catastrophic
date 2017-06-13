#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from copy import deepcopy

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class Model: 
    def __init__(self, x, y_, weights = 50):
        
        
        self.weights = weights
        
        in_dim = int(x.get_shape()[1])
        out_dim = int(y_.get_shape()[1])
        
        self.x = x
        
        
        W1 = weight_variable([in_dim, weights])
        b1 = bias_variable([weights])
        
        W2 = weight_variable([weights, out_dim])
        b2 = bias_variable([out_dim])
        
        self.var_list = [W1, b1, W2, b2]
        self.expandable_weights = [[W1], [b1], [W2]]
        
        self.h1 = [tf.nn.relu(tf.matmul(x,self.expandable_weights[0][0]) + self.expandable_weights[1][0])]
        
        self.y = [tf.matmul(self.h1[0],self.expandable_weights[2][0]) + self.var_list[3]]
                          
        
        
        # vanilla single-task loss
        self.cross_entropy = [tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y[len(self.y) - 1]))]
        self.set_vanilla_loss()
        
        self.correct_prediction = [tf.equal(tf.argmax(self.y[len(self.y) - 1],1), tf.argmax(y_,1))]
        self.accuracy = [tf.reduce_mean(tf.cast(self.correct_prediction[len(self.correct_prediction) - 1], tf.float32))]
        
    def compute_fisher(self, imgset, sess, num_samples=200):
        # computer Fisher information for each parameter

        # initialize Fisher information for most recent task
        self.F_accum = []
        for v in range(len(self.var_list)):
            self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))

        # sampling a random class from softmax
        probs = tf.nn.softmax(self.y[len(self.y) - 1])
        class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

            
        tensorGraphCreate = tf.gradients(tf.log(probs[0,class_ind]), self.var_list)
        
        for i in range(num_samples):
            # select random input image
            im_ind = np.random.randint(imgset.shape[0])
            # compute first-order derivatives
            ders = sess.run(tensorGraphCreate, feed_dict={self.x: imgset[im_ind:im_ind+1]})
            # square the derivatives and add to total
            for v in range(len(self.F_accum)):
                self.F_accum[v] += np.square(ders[v])
            

        # divide totals by number of samples
        for v in range(len(self.F_accum)):
            self.F_accum[v] /= num_samples
                        
    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []
        
        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())
            
    def restore(self, sess):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.star_vars)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))

    def set_vanilla_loss(self):
        for i in range(len(self.cross_entropy)):
            self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy[i])
        
    def update_ewc_loss(self, lam):
        # elastic weight consolidation
        # lam is weighting for previous task(s) constraints

        if not hasattr(self, "ewc_loss"):
            self.ewc_loss = self.cross_entropy

        for v in range(len(self.star_vars)):
            self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.ewc_loss)
    
    def expand(self, x, y_, sess, weights):
        
        in_dim = int(x.get_shape()[1])
        out_dim = int(y_.get_shape()[1])
            
        W1_expand = weight_variable([in_dim, weights])
        b1_expand = bias_variable([weights])
            
        W2_expand = weight_variable([weights, out_dim])
        
        new_vars = [W1_expand, b1_expand, W2_expand]
        
        for tensorVar in new_vars:
            self.var_list.append(tensorVar) 
        
        for index in range(len(self.expandable_weights)):
            self.expandable_weights[index].append(new_vars[index])
            
        
        self.h1.append(tf.nn.relu(tf.matmul(x,tf.concat(self.expandable_weights[0], 1)) + tf.concat(self.expandable_weights[1], 0)))
        self.y.append(tf.matmul(self.h1[len(self.h1) - 1], tf.concat(self.expandable_weights[2], 0)) + self.var_list[3])    
        
        self.cross_entropy.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y[len(self.y) - 1])))
        self.correct_prediction.append(tf.equal(tf.argmax(self.y[len(self.y) - 1],1), tf.argmax(y_,1)))
        self.accuracy.append(tf.reduce_mean(tf.cast(self.correct_prediction[len(self.correct_prediction) - 1], tf.float32)))
        
            
        