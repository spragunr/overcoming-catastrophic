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
    def __init__(self, x, y_, weights, layers):
        
        in_dim = int(x.get_shape()[1])
        out_dim = int(y_.get_shape()[1])
        
        self.x = x
        
        self.var_list = []
        
        self.var_list.append(weight_variable([in_dim, weights]))
        self.var_list.append(bias_variable([weights]))
        
        for layer in range(layers - 1):
           self.var_list.append(weight_variable([weights, weights]))
           self.var_list.append(bias_variable([weights]))
        
        self.var_list.append(weight_variable([weights, out_dim]))
        self.var_list.append(bias_variable([out_dim]))
        
        self.architecture = []
        self.architecture.append(x)
        self.error_sum_array = [0]
        count = 0
        
        for i in range(layers):
            self.architecture.append(tf.nn.relu(tf.matmul(self.architecture[i], self.var_list[i + count]) + self.var_list[i + count + 1]))
            count += 1            
       
        self.architecture.append(tf.matmul(self.architecture[len(self.architecture) - 1], self.var_list[len(self.var_list) - 2]) + self.var_list[len(self.var_list) - 1])
        
        # vanilla single-task loss
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.architecture[len(self.architecture) - 1]))
        self.set_vanilla_loss()
        
        #the variables below are lists because of the expand function (see below)
        self.correct_prediction = tf.equal(tf.argmax(self.architecture[len(self.architecture) - 1],1), tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
    def compute_fisher(self, imgset, sess, F_archives, num_samples=200):
        
        # initialize Fisher information for most recent task
        self.F_accum = []
        for v in range(len(self.var_list)):
            self.F_accum.append(np.zeros(self.var_list[v].eval().shape))

        # sampling a random class from softmax
        probs = tf.nn.softmax(self.architecture[len(self.architecture) - 1])
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
            
    def restore(self, sess, expanding=False):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            if expanding == True:
                expanded_vars = []
                
                for model_var in range(len(self.var_list) - 1):
                    if model_var % 2 == 0:
                        expanded_vars.append(weight_variable(list(self.var_list[model_var].eval().shape)))
                    else:
                        expanded_vars.append(bias_variable(list(self.var_list[model_var].eval().shape)))
                
                sess.run(tf.variables_initializer(expanded_vars))
                
                
                for ex_var in range(len(expanded_vars)):
                    axis = -1
                    both_axes = False
                    if ex_var == 0:
                        axis = 1
                    elif ex_var == len(expanded_vars) - 1:
                        axis = 0
                    elif ex_var % 2 != 0:
                        axis = 0
                    else: 
                        both_axes = True
                
                    if both_axes == False:    
                        sess.run(tf.assign(self.var_list[ex_var], tf.concat([self.var_list[ex_var], expanded_vars[ex_var]], axis), validate_shape=False))
                    else:   
                        sess.run(tf.assign(self.var_list[ex_var], tf.concat([self.var_list[ex_var], expanded_vars[ex_var]], 0), validate_shape=False))
                        sess.run(tf.assign(self.var_list[ex_var], tf.concat([self.var_list[ex_var], expanded_vars[ex_var]], 1), validate_shape=False))
                
                
                for saved_var in range(len(self.star_vars) - 1):
                    axis = -1
                    both_axes = False
                    if saved_var == 0:
                        axis = 1
                    elif saved_var == len(self.star_vars) - 2:
                        axis = 0
                    elif saved_var % 2 != 0:
                        axis = 0
                    else: 
                        both_axes = True
                        
                    if both_axes == False:    
                        self.star_vars[saved_var] = np.append(self.star_vars[saved_var], expanded_vars[saved_var].eval(), axis = axis)
                    else:   
                        self.star_vars[saved_var] = np.append(self.star_vars[saved_var], expanded_vars[saved_var].eval(), axis = 0)
                        self.star_vars[saved_var] = np.append(self.star_vars[saved_var], expanded_vars[saved_var].eval(), axis = 1)
                
            for v in range(len(self.star_vars)):
                sess.run(tf.assign(self.var_list[v], self.star_vars[v]))
                
    def set_vanilla_loss(self):
        
        self.train_step = tf.train.AdamOptimizer().minimize(self.cross_entropy)
    
    def update_ewc_loss(self, lam, F_archives, dim_dict):
        # elastic weight consolidation
        # lam is weighting for previous task(s) constraints
        self.ewc_loss = self.cross_entropy
        expansion = 100

        penalty = tf.constant(0.0)
        if len(F_archives) <= expansion - 2:
            for F_matrix in range(len(F_archives)):
                for net_var in range(len(self.var_list)):
                    penalty += tf.reduce_sum(tf.multiply(F_archives[F_matrix][net_var].astype(np.float32),tf.square(self.var_list[net_var] - self.star_vars[net_var])))
            self.ewc_loss += (lam/2.0) * penalty
                    
        else:
            for F_matrix_index in range(expansion - 2):
                for tensor in range(len(self.var_list) - 1):
                    if tensor % 2 == 0:
                        self.ewc_loss += (lam/2) * \
                        tf.reduce_sum(tf.multiply(F_archives[F_matrix_index][tensor].astype(np.float32),tf.square(tf.slice(self.var_list[tensor],
                                                                                                                           [0,0], dim_dict[expansion - 1][tensor]) -
                                                                                                                  tf.slice(self.star_vars[tensor], [0,0],
                                                                                                                           dim_dict[expansion - 1][tensor]))))
                        
                    else:
                        self.ewc_loss += (lam/2) * \
                        tf.reduce_sum(tf.multiply(F_archives[F_matrix_index][tensor].astype(np.float32),tf.square(tf.slice(self.var_list[tensor],
                                                                                                                           [0], dim_dict[expansion - 1][tensor]) -
                                                                                                                  tf.slice(self.star_vars[tensor], [0],
                                                                                                                           dim_dict[expansion - 1][tensor]))))
                        
                self.ewc_loss += (lam/2) * \
                tf.reduce_sum(tf.multiply(F_archives[F_matrix_index][len(F_archives[F_matrix_index])
                                                                     -
                                                                     1].astype(np.float32),tf.square(self.var_list[len(self.var_list)
                                                                                                                   - 1] - self.star_vars[len(self.star_vars) - 1])))
                
            for v in range(len(self.var_list)):
                for F_matrix_expanded in range(expansion - 2, len(F_archives)):
                    self.ewc_loss += (lam/2) * \
                    tf.reduce_sum(tf.multiply(F_archives[F_matrix_expanded][v].astype(np.float32),tf.square(self.var_list[v]
                                                                                                            - self.star_vars[v])))          
        
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.ewc_loss)
        self.penalty = penalty
