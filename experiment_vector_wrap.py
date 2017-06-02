
# coding: utf-8


import tensorflow as tf
import numpy as np
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data


# In[3]:

import matplotlib.pyplot as plt
from IPython import display


# In[4]:

# import class Model
from model2 import Model


# In[5]:

# mnist imshow convenience function
# input is a 1D array of length 784
def mnist_imshow(img):
    plt.imshow(img.reshape([28,28]), cmap="gray")
    plt.axis('off')
    
def calc_permutation_vectors(pixelCount, dist_horiz, dist_vert):
    delta_x = np.random.random_integers(-dist_horiz, dist_horiz, pixelCount)
    delta_y = np.random.random_integers(-dist_vert, dist_vert, pixelCount)
    return delta_x, delta_y

def permute_mnist_vector_wrap(mnist, delta_x, delta_y):
    pixels = np.array(range(mnist.train.images.shape[1]))
    deltaNum = 0
    dim_array = np.sqrt([len(pixels)])
    dim = dim_array[0]
    for pixel in range(len(pixels)):
        new_pix_loc = pixel
        x_movement = delta_x[deltaNum]
        if ((pixel + x_movement) < len(pixels) and (pixel + x_movement) >= 0):
            new_pix_loc += x_movement
        else:
            if (x_movement > 0):
                while (x_movement > 0):
                    if (new_pix_loc + 1) % dim == 0:
                        new_pix_loc -= (dim - 1)
                    else: 
                        new_pix_loc += 1
                    x_movement -= 1
            elif (x_movement < 0):
                while (x_movement < 0):
                    if new_pix_loc % dim == 0:
                        new_pix_loc += (dim - 1)
                    else: 
                        new_pix_loc -= 1
                    x_movement += 1
        y_movement = delta_y[deltaNum]
        if ((new_pix_loc + (dim * y_movement)) < len(pixels) and (new_pix_loc + (dim * y_movement)) >= 0):
            new_pix_loc += (dim * y_movement)
        else:
            if (y_movement > 0):
                while (y_movement > 0):
                    if (new_pix_loc + dim) > (len(pixels) - 1):
                        new_pix_loc -= (dim * (dim - 1))
                    else: 
                        new_pix_loc += dim
                    y_movement -= 1
            elif (y_movement < 0):
                while (y_movement < 0):
                    if (new_pix_loc - dim) < 0:
                        new_pix_loc += (dim * (dim - 1))
                    else: 
                        new_pix_loc -= dim
                    y_movement += 1
                
        pixels[pixel] = pixels[new_pix_loc]
        deltaNum += 1
        
    mnist2 = deepcopy(mnist)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(mnist2, set_name) # shallow copy
        this_set._images = np.transpose(np.array([this_set.images[:,c] for c in pixels]))
    return mnist2

# In[6]:

# classification accuracy plotting
def plot_test_acc(plot_handles):
    plt.legend(handles=plot_handles, loc="center right")
    plt.xlabel("Iterations")
    plt.ylabel("Test Accuracy")
    plt.ylim(0,1)
    display.display(plt.gcf())
    display.clear_output(wait=True)
    
# train/compare vanilla sgd and ewc
acc_sgd = []
acc_ewc = []
figNum = 1
def train_task(model, num_iter, disp_freq, trainset, testsets, x, y_, lams=[0]):
    global acc_sgd
    global acc_ewc
    global figNum
    
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
                    test_accs[iter/disp_freq][task] = model.accuracy.eval(feed_dict=feed_dict)
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
    plt.plot(sgdList, label="sgd")
    plt.plot(ewcList, label="ewc")
    plt.xlabel('tasks')
    plt.ylabel('average accuracy')
    plt.axis([1, 11, 0, 1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    figNum += 1
    

# In[7]:

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# In[8]:

sess = tf.InteractiveSession()


# In[9]:

# define input and target placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# In[10]:

# instantiate new model
model = Model(x, y_) # simple 2-layer network


# In[11]:

# initialize variables
sess.run(tf.global_variables_initializer())


# #### train on task A, test on task A

# In[12]:

# training 1st task
train_task(model, 800, 20, mnist, [mnist], x, y_, lams=[0])


# In[13]:

# Fisher information
model.compute_fisher(mnist.validation.images, sess, num_samples=200, plot_diffs=True) # use validation set for Fisher computation


# In[14]:

F_row_mean = np.mean(model.F_accum[0], 1)


# #### train on task B, test on tasks A and B

# In[15]:
permutation_vector_x, permutation_vector_y = calc_permutation_vectors(mnist.train.images.shape[1], 1, 1)
# permuting mnist for 2nd task
mnist2 = permute_mnist_vector_wrap(mnist, permutation_vector_x, permutation_vector_y)
# permuting mnist for 2nd task




# In[16]:

# save current optimal weights
model.star()

mnistList = [mnist, mnist2]
mnist = mnist2
for i in range(1, 15):
    train_task(model, 800, 20, mnist, mnistList, x, y_, lams=[0, 15])
    model.compute_fisher(mnist.validation.images, sess, num_samples=200, plot_diffs=True)
    F_row_mean = np.mean(model.F_accum[0], 1)
    permutation_vector_x, permutation_vector_y = calc_permutation_vectors(mnist.train.images.shape[1], 1, 1)
    mnist = permute_mnist_vector_wrap(mnist, permutation_vector_x, permutation_vector_y)
    mnistList.append(mnist)
    model.star()


# In[ ]:



