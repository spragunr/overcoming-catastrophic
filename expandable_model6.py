#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

weight_keeper = []
stars = []

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class Model: 
    def __init__(self, x, y_, weights = 20, expand = False):
        
        global weight_keeper
        global stars
        
        self.weights = weights
        
        in_dim = int(x.get_shape()[1])
        out_dim = int(y_.get_shape()[1])
        
        self.x = x
        
        
        W1 = weight_variable([in_dim, weights])
        b1 = bias_variable([weights])
        
        W2 = weight_variable([weights, out_dim])
        b2 = bias_variable([out_dim])
        
        self.var_list = [W1, b1, W2, b2]
        
        if expand == True:
            sess = tf.InteractiveSession()   
            for weightVar in range(len(weight_keeper)):
                weight_keeper[weightVar] = tf.convert_to_tensor(weight_keeper[weightVar], dtype = tf.float32)
                weight_keeper[weightVar] = tf.Variable(weight_keeper[weightVar])
            sess.run(tf.global_variables_initializer())
            for networkWeight in range(len(self.var_list)):
                tf.assign(self.var_list[networkWeight], weight_keeper[networkWeight])
                            
            self.star_vars = []
            for star in range(len(stars)):
                self.star_vars.append(stars[star])
        
        h1 = tf.nn.relu(tf.matmul(x,W1) + b1)
        
        self.y = tf.matmul(h1,W2) + b2
                          
        
        
        # vanilla single-task loss
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y))
        self.set_vanilla_loss()
        
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    def compute_fisher(self, imgset, sess, num_samples=200):
        # computer Fisher information for each parameter

        # initialize Fisher information for most recent task
        self.F_accum = []
        for v in range(len(self.var_list)):
            self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))

        # sampling a random class from softmax
        probs = tf.nn.softmax(self.y)
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
        global stars
        stars = []
        self.star_vars = []
        
        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())
            stars.append(self.var_list[v].eval())     
    def restore(self, sess):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))

    def set_vanilla_loss(self):
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)
        
    def update_ewc_loss(self, lam):
        # elastic weight consolidation
        # lam is weighting for previous task(s) constraints

        if not hasattr(self, "ewc_loss"):
            self.ewc_loss = self.cross_entropy

        for v in range(len(self.var_list)):
            self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.ewc_loss)
    
    def save_weights(self, x, y_):
        
        global weight_keeper
        weight_keeper = []
        
        in_dim = int(x.get_shape()[1])
        out_dim = int(y_.get_shape()[1])
            
        W1_expand = weight_variable([in_dim, self.weights])
        b1_expand = bias_variable([self.weights])
            
        W2_expand = weight_variable([self.weights, out_dim])
            
        expansion_var_list = [W1_expand, b1_expand, W2_expand]
        sess = tf.InteractiveSession()
        sess.run(tf.variables_initializer(self.var_list))
        sess.run(tf.variables_initializer(expansion_var_list))
                
        for var in range(len(self.var_list)):
            if var == 0:
                new_weights = np.append(self.var_list[var].eval(), expansion_var_list[var].eval(), axis = 1)
            elif var == 2:
                new_weights = np.append(self.var_list[var].eval(), expansion_var_list[var].eval(), axis = 0)
            elif var == 1:
                new_weights = np.append(self.var_list[var].eval(), expansion_var_list[var].eval(), axis = 0)
            else:
                new_weights = self.var_list[var].eval()
            
            weight_keeper.append(new_weights)
            
            self.var_list[var] = tf.convert_to_tensor(new_weights)
            self.var_list[var] = tf.Variable(self.var_list[var])
        
        sess.run(tf.variables_initializer(self.var_list))
        
        self.star()
            
            
        