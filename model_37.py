#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  
    return tf.Variable(initial, validate_shape=False)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, validate_shape=False)

class Model: 
    def __init__(self, x, y_, weights = 30):
        
        in_dim = int(x.get_shape()[1])
        out_dim = int(y_.get_shape()[1])
        
        self.x = x
        
        self.W1 = weight_variable([in_dim, weights])
        self.b1 = bias_variable([weights])
        
        self.W2 = weight_variable([weights, out_dim])
        self.b2 = bias_variable([out_dim])
        
        self.var_list = [self.W1, self.b1, self.W2, self.b2]
        
        #middle layer
        self.h1 = tf.nn.relu(tf.matmul(x,self.var_list[0]) + self.var_list[1])
        
        #output layer
        self.y = tf.matmul(self.h1,self.var_list[2]) + self.var_list[3] 
                          
        
        
        # vanilla single-task loss
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y))
        self.set_vanilla_loss()
        
        #the variables below are lists because of the expand function (see below)
        self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
    def compute_fisher(self, imgset, sess, F_archives, num_samples=200):
        
        # initialize Fisher information for most recent task
        self.F_accum = []
        for v in range(len(self.var_list)):
            self.F_accum.append(np.zeros(self.var_list[v].eval().shape))

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
        
        F_archives.append(self.F_accum)
    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []
        
        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())
            
    def restore(self, sess, weights, x, y_, expanding=False):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            if expanding == True:
                in_dim = int(x.get_shape()[1])
                out_dim = int(y_.get_shape()[1])
                
                expand_W1 = weight_variable([in_dim, weights])
                expand_b1 = bias_variable([weights])
                
                expand_W2 = weight_variable([weights, out_dim])
                
                sess.run(tf.variables_initializer([expand_W1, expand_b1, expand_W2]))
                
                sess.run(tf.assign(self.var_list[0], tf.concat([self.var_list[0], expand_W1], 1), validate_shape=False))
                sess.run(tf.assign(self.var_list[1], tf.concat([self.var_list[1], expand_b1], 0), validate_shape=False))
                sess.run(tf.assign(self.var_list[2], tf.concat([self.var_list[2], expand_W2], 0), validate_shape=False))
                
                
                self.star_vars[0] = np.append(self.star_vars[0], expand_W1.eval(), axis = 1)
                self.star_vars[1] = np.append(self.star_vars[1], expand_b1.eval(), axis = 0)
                self.star_vars[2] = np.append(self.star_vars[2], expand_W2.eval(), axis = 0)
            
                self.h1 = tf.nn.relu(tf.matmul(x,self.var_list[0]) + self.var_list[1])
                self.y = tf.matmul(self.h1,self.var_list[2]) + self.var_list[3] 
                
                
                self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y))
                self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            
            for v in range(len(self.star_vars)):
                sess.run(tf.assign(self.var_list[v], self.star_vars[v]))
                
    def set_vanilla_loss(self):
        
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)
    
    def update_ewc_loss(self, lam, F_archives):
        # elastic weight consolidation
        # lam is weighting for previous task(s) constraints
        self.ewc_loss = self.cross_entropy
        
        if len(F_archives) <= 2:
            for F_matrix in range(len(F_archives)):
                self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(F_archives[F_matrix][0].astype(np.float32),tf.square(tf.slice(self.var_list[0], [0,0], [784,30]) - tf.slice(self.star_vars[0], [0,0], [784,30]))))
                self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(F_archives[F_matrix][1].astype(np.float32),tf.square(tf.slice(self.var_list[1], [0], [30]) - tf.slice(self.star_vars[1], [0], [30]))))
                self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(F_archives[F_matrix][2].astype(np.float32),tf.square(tf.slice(self.var_list[2], [0,0], [30,10]) - tf.slice(self.star_vars[2], [0,0], [30,10]))))
                self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(F_archives[F_matrix][3].astype(np.float32),tf.square(self.var_list[3] - self.star_vars[3])))
        else:
            for F_matrix in range(2):
                self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(F_archives[F_matrix][0].astype(np.float32),tf.square(tf.slice(self.var_list[0], [0,0], [784,30]) - tf.slice(self.star_vars[0], [0,0], [784,30]))))
                self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(F_archives[F_matrix][1].astype(np.float32),tf.square(tf.slice(self.var_list[1], [0], [30]) - tf.slice(self.star_vars[1], [0], [30]))))
                self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(F_archives[F_matrix][2].astype(np.float32),tf.square(tf.slice(self.var_list[2], [0,0], [30,10]) - tf.slice(self.star_vars[2], [0,0], [30,10]))))
                self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(F_archives[F_matrix][3].astype(np.float32),tf.square(self.var_list[3] - self.star_vars[3])))
            for v in range(len(self.var_list)):
                for F_matrix_expanded in range(2, len(F_archives)):
                    self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(F_archives[F_matrix_expanded][v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.ewc_loss)