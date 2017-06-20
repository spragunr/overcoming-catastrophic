
# coding: utf-8


import tensorflow as tf
import numpy as np
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt

# import class Model
from model_09 import Model

# return a new mnist dataset w/ pixels randomly permuted
def permute_mnist(mnist):
    perm_inds = range(mnist.train.images.shape[1])
    np.random.shuffle(perm_inds)
    mnist2 = deepcopy(mnist)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(mnist2, set_name) # shallow copy
        this_set._images = np.transpose(np.array([this_set.images[:,c] for c in perm_inds]))
    return mnist2

# train/compare vanilla sgd and ewc
acc_sgd = []
acc_ewc = []
final_means = [0]
figNum = 1
def train_task(model, num_iter, disp_freq, trainset, testsets, x, y_, lams=[0]):
    global acc_sgd
    global acc_ewc
    global figNum
    global final_means
    for l in range(len(lams)):
        # lams[l] sets weight on old task(s)
        model.restore(sess) # reassign optimal weights from previous training session
        if(lams[l] == 0):
            model.set_vanilla_loss()
            acc_array = acc_sgd
        else:
            model.update_ewc_loss(lams[l])
            acc_array = acc_ewc
            
        #initialize test accuracy array holding all tasks (for given training methodology)
        test_accs = np.zeros((40, len(testsets)))
        
        # train on current task
        for iter in range(num_iter):
            batch = trainset.train.next_batch(100) 
            model.train_step.run(feed_dict={x: batch[0], y_: batch[1]})
            if iter % disp_freq == 0:
                for task in range(len(testsets)):
                    feed_dict={x: testsets[task].test.images, y_: testsets[task].test.labels}
                    if task<4:
                        if expansion == True:
                            model.alt_h1.append(tf.nn.relu(tf.matmul(x, tf.slice(model.expandable_weights[0][0], [0,0], [784,50])) + tf.slice(model.expandable_weights[1][0], [0], [50])))
                            model.alt_y.append(tf.matmul(model.alt_h1[len(model.alt_h1) - 1], tf.slice(model.expandable_weights[2][0], [0,0], [50, 10])) + model.expandable_weights[3][0])
                            
                            model.alt_correct_prediction.append(tf.equal(tf.argmax(model.alt_y[len(model.alt_y) - 1],1), tf.argmax(y_,1)))
                            model.alt_accuracy.append(tf.reduce_mean(tf.cast(model.alt_correct_prediction[len(model.alt_correct_prediction) - 1], tf.float32)))
                            
                            test_accs[iter/disp_freq][task] = model.alt_accuracy[0].eval(feed_dict=feed_dict)
                        else:
                            test_accs[iter/disp_freq][task] = model.accuracy[0].eval(feed_dict=feed_dict)
                    else:
                        test_accs[iter/disp_freq][task] = model.accuracy[1].eval(feed_dict=feed_dict)
                  
        accSum = 0
        for taskColumn in range(len(test_accs[0])):
            accSum += test_accs[39, taskColumn]
        acc_array.append(accSum / len(test_accs[0]))
        if len(lams) == 1:
            acc_ewc.append(accSum / len(test_accs[0]))
            
    sgdList = [0]
    ewcList = [0]
    for listIndex in range(len(acc_sgd)):
        sgdList.append(acc_sgd[listIndex])
        ewcList.append(acc_ewc[listIndex])
    
    

    
    plt.figure(figNum)
    plt.subplot(121)
    plt.plot(sgdList, label="sgd")
    plt.plot(ewcList, label="ewc")
    plt.xlabel('tasks')
    plt.ylabel('average accuracy')
    plt.axis([1, 11, 0, 1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
 
    if hasattr(model, "F_accum"):
        f = model.F_accum
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
    plt.show()
    figNum += 1

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)



# define input and target placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])



# instantiate new model
model = Model(x, y_) # simple 2-layer network

# In[11]:
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
file_writer = tf.summary.FileWriter('/home/andrew/REU/overcoming-catastrophic-master', sess.graph)

# #### train on task A, test on task A
expansion = False
# training 1st task
train_task(model, 800, 20, mnist, [mnist], x, y_, lams=[0])
    
    # Fisher information
model.compute_fisher(mnist.validation.images, sess, num_samples=200) # use validation set for Fisher computation
    
mnist2 = permute_mnist(mnist)
    
    # save current optimal weights
model.star()

mnistList = [mnist, mnist2]
mnist = mnist2
tasks = 1 
cur_weights = model.weights
for i in range(1, 15):
    tasks += 1
    if tasks == 4:
        model.expand(x, y_, sess, cur_weights)
        expansion = True
        cur_weights *= 2
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        file_writer = tf.summary.FileWriter('/home/andrew/REU/overcoming-catastrophic-master', sess.graph)
        model.compute_fisher(mnist.validation.images, sess, num_samples=200)
    train_task(model, 800, 20, mnist, mnistList, x, y_, lams=[0, 15])
    model.compute_fisher(mnist.validation.images, sess, num_samples=200)
    mnist = permute_mnist(mnist)
    mnistList.append(mnist)
    model.star()


# In[ ]:



