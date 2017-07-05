# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data

from model_37 import Model





# return a new mnist dataset w/ pixels randomly permuted
def permute_mnist(mnist):
    perm_inds = range(mnist.train.images.shape[1])
    np.random.shuffle(perm_inds)
    mnist2 = deepcopy(mnist)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(mnist2, set_name) # shallow copy
        this_set._images = np.transpose(np.array(
                [this_set.images[:,c] for c in perm_inds]))
    return mnist2

# train/compare vanilla sgd and ewc
def train_task(model, sess, num_iter, disp_freq, trainset, testsets, x, y_, acc_sgd, 
               acc_ewc, final_means, figNum, F_archives, lams=[0], expanding=False):
    # lams[l] sets weight on old task(s)
    # l == 0 coincides with vanilla SGD training
    expanded_already = False
    for l in range(len(lams)):
        
        # if network has just expanded before this training run AND this is the
        # first time the training is occuring in the lams[l] loop,
        # restore the weights from old tasks to the first half of the new 
        # weights in the expanded (doubled capacity) network
        
        if expanding == True and expanded_already == False:
            model.restore(sess, cur_weights, x, y_, expanding=True)
            expanded_already = True
            
        # otherwise, simply restore the weights from the last model.star() call
        # to the entire tensors constituting the weights in existing network
        else:
            model.restore(sess, cur_weights, x, y_) 
        
        if(lams[l] == 0):
            model.set_vanilla_loss()
            acc_array = acc_sgd
        else:
            model.update_ewc_loss(lams[l], F_archives)
            acc_array = acc_ewc
            
        # initialize test accuracy array holding all tasks for given lams[l]
        test_accs = np.zeros((40, len(testsets)))
        
        # train on all tasks for given number of iterations
        for iter in range(num_iter):
            batch = trainset.train.next_batch(100) 
            model.train_step.run(feed_dict={x: batch[0], y_: batch[1]})
            # controls how frequently to save accuracy for display in graph
            if iter % disp_freq == 0:
                
                for task in range(len(testsets)):
                    feed_dict={x: testsets[task].test.images, y_: testsets[task].test.labels}
                    
                    if task < 3:
                        
                        alt_h1 = tf.nn.relu(tf.matmul(x, tf.slice(model.var_list[0], [0,0], [784,30])) + tf.slice(model.var_list[1], [0], [30]))
                        alt_y = tf.matmul(alt_h1, tf.slice(model.var_list[2], [0,0], [30, 10])) + model.var_list[3]
                        
                        alt_correct_prediction = tf.equal(tf.argmax(alt_y,1), tf.argmax(y_,1))
                        alt_accuracy = tf.reduce_mean(tf.cast(alt_correct_prediction, tf.float32))
                        
                        test_accs[iter/disp_freq][task] = alt_accuracy.eval(feed_dict=feed_dict)
                    
                    else:
                        test_accs[iter/disp_freq][task] = model.accuracy.eval(feed_dict=feed_dict)
                  
        #storing accuracy data for plotting
        accSum = 0
        
        #take the last accuracy reading for each task and add it to accSum
        for taskColumn in range(len(test_accs[0])):
            accSum += test_accs[iter/disp_freq - 1, taskColumn]
        
        #append the average of all of the last task accuracy readings
        #to the acc_array for SGD or EWC depending on loop 
        #*this is what you see in the graph*
        acc_array.append(accSum / len(test_accs[0]))
        
        #if this is the first training run, simply set the first task EWC
        #accuracy equal to the first task accuracy for SGD
        if len(lams) == 1:
            acc_ewc.append(accSum / len(test_accs[0]))
    
    #this merely shifts the data in acc_sgd and acc_ewc forward one index
    #so that it aligns with task number in graph (starting at 1 task)        
    sgdList = [0]
    ewcList = [0]
    for listIndex in range(len(acc_sgd)):
        sgdList.append(acc_sgd[listIndex])
        ewcList.append(acc_ewc[listIndex])
    
    #plot average accuracy over all taks for both SGD and EWC
    plt.figure(figNum[0])
    plt.subplot(111)
    plt.plot(sgdList, label="sgd")
    plt.plot(ewcList, label="ewc")
    plt.xlabel('tasks')
    plt.ylabel('average accuracy')
    plt.axis([1, 15, 0, 1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
 
    """
    #plotting Fisher data if available
    if hasattr(model, "F_accum"):
        means_list = []
        for diag in range(len(model.F_accum)):
            diag_sum = 0
            elementCount = 0
            fim_diagonal = model.F_accum[diag]
            for row in range(len(fim_diagonal)):
                if diag % 2 == 0:
                    for col in range(len(fim_diagonal[row])):
                        diag_sum += fim_diagonal[row][col]
                        elementCount += 1
                else:
                    diag_sum += fim_diagonal[row]
                    elementCount += 1
            means_list.append(diag_sum / float(elementCount))
            
            
        final_means.append(final_means[len(final_means) - 1] + np.sum(means_list)/ float(len(means_list)))
        plt.subplot(122)
        plt.plot(final_means, label="Fisher")
        plt.xlabel('tasks')
        plt.ylabel('Average Fisher Information Diagonal Mean')
        plt.axis([1, 11, 0, 0.3])
    """
    plt.savefig("last_figure_updated_expanded_revised.png")
    figNum[0] += 1

#initial MNIST data read-in
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# define input and target placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#construct a new Model
model = Model(x, y_) 

#install sess as the TensorFlow default session and initialize all variables 
#in the model
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


#holds all of the mnist permutations, including the original dataset
mnistList = [mnist]

#used to track task count in training/testing loop 
tasks = 0 

#network initial node count 
cur_weights = 30

acc_sgd = []
acc_ewc = []

final_means = [0]

figNum = [1]
F_archives = []
for i in range(15):
    tasks += 1
    if tasks == 1:
        train_task(model, sess, 800, 20, mnist, mnistList, x, y_, acc_sgd, acc_ewc, final_means, figNum, F_archives, lams=[0])
    elif tasks == 4:    
        model.F_accum[0] = np.append(model.F_accum[0], np.zeros((784, 30)), axis=1)
        model.F_accum[1] = np.append(model.F_accum[1], np.zeros((30)), axis = 0)
        model.F_accum[2] = np.append(model.F_accum[2], np.zeros((30, 10)), axis = 0)
        F_archives[len(F_archives) - 1] = model.F_accum
        train_task(model, sess, 800, 20, mnist, mnistList, x, y_, acc_sgd, acc_ewc, final_means, figNum, F_archives, lams=[0, 15], expanding=True)
        cur_weights *= 2
    else:
        train_task(model, sess, 800, 20, mnist, mnistList, x, y_, acc_sgd, acc_ewc, final_means, figNum, F_archives, lams=[0, 15])
    model.compute_fisher(mnist.validation.images, sess, F_archives, num_samples=200)
    model.star()
    mnist = permute_mnist(mnist)
    mnistList.append(mnist)