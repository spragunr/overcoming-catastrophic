# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data

from model_52 import Model

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

def permute_mnist_random(mnist, percent):
    old_pixels = np.array(range(mnist.train.images.shape[1]))
    new_pixels = np.array(range(mnist.train.images.shape[1]))
    np.random.shuffle(new_pixels)
    
    locations = np.random.choice(range(len(old_pixels)), int(len(old_pixels) - (len(old_pixels) * float(percent))), replace=False)
    
    for pixel in locations:
        new_pixels[pixel] = old_pixels[pixel]
        
    mnist2 = deepcopy(mnist)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(mnist2, set_name) # shallow copy
        this_set._images = np.transpose(np.array([this_set.images[:,c] for c in new_pixels]))
    return mnist2


# train/compare vanilla sgd and ewc
def train_task(model, sess, num_iter, disp_freq, trainset, testsets, x, y_, acc_sgd, 
               acc_ewc, final_means, figNum, F_archives, tasks, lams=[0], expanding=False):
    # lams[l] sets weight on old task(s)
    # l == 0 coincides with vanilla SGD training
    
    expanded_already = False
    
    for l in range(len(lams)):
        
        # if network has just expanded before this training run AND this is the
        # first time the training is occuring in the lams[l] loop,
        # restore the weights from old tasks to the first half of the new 
        # weights in the expanded (doubled capacity) network
        
        if expanding == True and expanded_already == False:
            model.restore(sess, expanding=True)
            expanded_already = True
            
        # otherwise, simply restore the weights from the last model.star() call
        # to the entire tensors constituting the weights in existing network
        else:
            model.restore(sess) 
        
        dims_list = []
        for var in range(len(model.var_list)):
            dims_list.append(list(model.var_list[var].eval().shape))
        
        dim_dict[tasks] = dims_list
        
        if(lams[l] == 0):
            model.set_vanilla_loss()
            acc_array = acc_sgd
        else:
            model.update_ewc_loss(lams[l], F_archives, dim_dict)
            acc_array = acc_ewc
            
        # initialize test accuracy array holding all tasks for given lams[l]
        test_accs = np.zeros((40, len(testsets)))
        
        # train on all tasks for given number of iterations
        for iteration in range(num_iter):
            batch = trainset.train.next_batch(100) 
            model.train_step.run(feed_dict={x: batch[0], y_: batch[1]})
            if hasattr(model, 'penalty'):
                print model.penalty.eval(feed_dict={x: batch[0], y_: batch[1]}) 
            # controls how frequently to save accuracy for display in graph
            if iteration % disp_freq == 0:
                
                for task in range(len(testsets)):
                    
                    feed_dict={x: testsets[task].test.images, y_: testsets[task].test.labels}
                    """
                    slice_vars = []
                    slicer = dim_dict[task + 1]
                    
                    for tensor in range(len(model.var_list) - 1):
                        if tensor % 2 == 0:
                            slice_vars.append(tf.slice(model.var_list[tensor], [0,0], slicer[tensor]))
                        else:
                            slice_vars.append(tf.slice(model.var_list[tensor], [0], slicer[tensor]))
                    
                    alt_architecture = []       
                    alt_architecture.append(x)
                    
                    count = 0
        
                    for j in range(len(model.architecture) - 2):
                        alt_architecture.append(tf.nn.relu(tf.matmul(alt_architecture[j], slice_vars[j + count]) + slice_vars[j + count + 1]))
                        count += 1
                    
                    alt_architecture.append(tf.matmul(alt_architecture[len(alt_architecture) - 1], slice_vars[len(slice_vars) - 1]) + model.var_list[len(model.var_list) - 1])
                        
                    alt_correct_prediction = tf.equal(tf.argmax(alt_architecture[len(alt_architecture) - 1],1), tf.argmax(y_,1))
                    alt_accuracy = tf.reduce_mean(tf.cast(alt_correct_prediction, tf.float32))
                        
                    test_accs[iter/disp_freq][task] = alt_accuracy.eval(feed_dict=feed_dict)
                    
                    """
                    test_accs[iteration/disp_freq][task] = model.accuracy.eval(feed_dict=feed_dict)
        
        for taskNumber in range(len(testsets)):
            dataFile.create_dataset('count {} task {} lambda {} run {}'.format(str(len(testsets)), str(taskNumber + 1), str(lams[l]), str(run + 1)), data=test_accs[:,taskNumber])
            
        #storing accuracy data for plotting
        accSum = 0
        accSumOld = 0
        #take the last accuracy reading for each task and add it to accSum
        for taskColumn in range(len(test_accs[0])):
            accSum += test_accs[iteration/disp_freq - 1, taskColumn]
        
        for taskColumnOld in range(len(test_accs[0]) - 1):
            accSumOld += test_accs[iteration/disp_freq - 1, taskColumn]
        if len(testsets) > 1:
            avg_acc_old_tasks.append(accSumOld / float(len(test_accs[0]) - 1))
        
        
        
        
        acc_most_recent_task.append(test_accs[len(test_accs) - 1][len(test_accs[len(test_accs) - 1]) - 1])
        
        
            
        #append the average of all of the last task accuracy readings
        #to the acc_array for SGD or EWC depending on loop 
        #*this is what you see in the graph*
        acc_array.append(accSum / float(len(test_accs[0])))
        
        #if this is the first training run, simply set the first task EWC
        #accuracy equal to the first task accuracy for SGD
        if len(lams) == 1:
            acc_ewc.append(accSum / float(len(test_accs[0])))
    
    #this merely shifts the data in acc_sgd and acc_ewc forward one index
    #so that it aligns with task number in graph (starting at 1 task)        
    sgdList = [0]
    ewcList = [0]
    for listIndex in range(len(acc_sgd)):
        sgdList.append(acc_sgd[listIndex])
        ewcList.append(acc_ewc[listIndex])
    """
    #plot average accuracy over all taks for both SGD and EWC
    plt.figure(num=figNum[0], figsize=(20,10))
    plot1 = plt.subplot(121)
    plot1.set_title("layers: {}  weights per layer: {}  lambda: {}  percent permutation: {} permutation type: {} run: {}".format(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], str(run + 1)))
    plt.plot(sgdList, label="sgd")
    plt.plot(ewcList, label="ewc")
    plt.xlabel('tasks')
    plt.ylabel('average accuracy')
    plt.axis([1, 25, 0.75, 0.95])
    plt.legend(loc=3)
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
            
        fisher_most_recent.append(np.sum(means_list)/ float(len(means_list)))
        
        final_means.append(final_means[len(final_means) - 1] + np.sum(means_list)/ float(len(means_list)))
        
        
        """
        plot2 = plt.subplot(122)
        plot2.set_title("layers: {}  weights per layer: {}  lambda: {}  percent permutation: {} permutation type: {} run: {}".format(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], str(run + 1)))
        plt.plot(final_means, label="Fisher")
        plt.xlabel('tasks')
        plt.ylabel('Average Fisher Information Diagonal Mean')
        plt.axis([1, 25, 0, 0.75])
        
    plt.savefig(sys.argv[6])
    plt.close()
    figNum[0] += 1
    """
    
    if len(testsets) > 1 and avg_acc_old_tasks[len(avg_acc_old_tasks) - 1] < 0.1:
        dataFile.create_dataset('average old task accuracy lambda {} run {}'.format(lams[l], str(run + 1)), data=avg_acc_old_tasks)
        dataFile.create_dataset('most recent task accuracy lambda {} run {}'.format(lams[l], str(run + 1)), data=acc_most_recent_task)
        dataFile.create_dataset('fisher sum run {}'.format(str(run + 1)), data=final_means)
        dataFile.create_dataset('most recent fisher run {}'.format(str(run + 1)), data=fisher_most_recent)
        run_over[0] = 1
        if run == int(int(sys.argv[8]) - 1):
            dataFile.close()
            sess.close()
            sys.exit()


dataFile = h5py.File(sys.argv[7], 'w')
dataFile.create_dataset('params', data=sys.argv[1:])
for run in range(int(sys.argv[8])):
    
    #initial MNIST data read-in
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    
    # define input and target placeholders
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    
    #network initial node count 
    cur_weights = sys.argv[2]
    
    #construct a new Model
    model = Model(x, y_, int(sys.argv[2]), int(sys.argv[1])) 
    
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    #install sess as the TensorFlow default session and initialize all variables 
    #in the model
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    
    
    #holds all of the mnist permutations, including the original dataset
    mnistList = [mnist]
    
    #used to track task count in training/testing loop 
    tasks = 0 
    
    
    
    acc_sgd = []
    acc_ewc = []
    
    final_means = [0]
    figNum = [1]
    F_archives = []
    dim_dict = {}
    lambdas = [0, float(sys.argv[3])]
    
    avg_acc_old_tasks = [0, 0]
    acc_most_recent_task = [0, 0]
    fisher_most_recent = [0]
    run_over = [0]
    for i in range(100):
        tasks += 1
        print(dim_dict)
        if tasks == 1:
            train_task(model, sess, 800, 20, mnistList[len(mnistList) - 1], mnistList, x, y_, acc_sgd, acc_ewc, final_means, figNum, F_archives, tasks, lams=[0])
        elif tasks == 100:    
            for FIM in range(len(model.F_accum) - 1):
                append_axis = -1
                both_axes = False
                
                if FIM == 0:
                    append_axis = 1
                elif FIM == len(model.F_accum) - 2:
                    append_axis = 0
                elif FIM % 2 != 0:
                    append_axis = 0
                else: 
                    both_axes = True
                
                if both_axes == False:
                    model.F_accum[FIM] = np.append(model.F_accum[FIM], np.zeros(tuple(model.F_accum[FIM].shape)), axis=append_axis)
                else:
                    model.F_accum[FIM] = np.append(model.F_accum[FIM], np.zeros(tuple(model.F_accum[FIM].shape)), axis=0)
                    model.F_accum[FIM] = np.append(model.F_accum[FIM], np.zeros(tuple(model.F_accum[FIM].shape)), axis=1)
                
            F_archives[len(F_archives) - 1] = model.F_accum
            train_task(model, sess, 800, 20, mnistList[len(mnistList) - 1], mnistList, x, y_, acc_sgd, acc_ewc, final_means, figNum, F_archives, tasks, lams=lambdas, expanding=True)
            cur_weights *= 2
        else:
            train_task(model, sess, 800, 20, mnistList[len(mnistList) - 1], mnistList, x, y_, acc_sgd, acc_ewc, final_means, figNum, F_archives, tasks, lams=lambdas)
        if run_over[0] == 1:
            sess.close()
            break
        model.compute_fisher(mnistList[len(mnistList) - 1].validation.images, sess, F_archives, num_samples=200)
        model.star()
        if sys.argv[5] == "travel":
            mnistList.append(permute_mnist_random(mnistList[len(mnistList) - 1], float(float(sys.argv[4])/100.0)))
        elif sys.argv[5] == "spread":
            mnistList.append(permute_mnist_random(mnistList[0], float(float(sys.argv[4])/100.0)))
        else:
            print("NO PERMUTATION METHOD SPECIFIED")
