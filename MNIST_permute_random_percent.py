#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:52:23 2017
"""

import numpy as np
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt

# mnist imshow convenience function
# input is a 1D array of length 784
def mnist_imshow(img):
    plt.imshow(img.reshape([28,28]), cmap="gray")
    plt.axis('off')

# return a new mnist dataset w/ pixels randomly permuted
"""
def permute_mnist(mnist):
    perm_inds = range(mnist.train.images.shape[1])
    for pixel in (perm_inds)
    np.random.shuffle(perm_inds)
    mnist2 = deepcopy(mnist)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(mnist2, set_name) # shallow copy
        this_set._images = np.transpose(np.array([this_set.images[:,c] for c in perm_inds]))
    return mnist2
"""
    
def permute_mnist_random(mnist, percent):
    pixels = np.array(range(mnist.train.images.shape[1]))
    permute = np.random.random_integers(0, len(pixels) - 1, size=len(pixels)/(1.0/(percent/2.0)))
    swap = np.random.random_integers(0, len(pixels) - 1, size=len(permute))
    listCounter = 0
    for pixel in permute:
        temp = pixels[swap[listCounter]]
        pixels[swap[listCounter]] = pixels[pixel]
        pixels[pixel] = temp
        listCounter += 1
    mnist2 = deepcopy(mnist)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(mnist2, set_name) # shallow copy
        this_set._images = np.transpose(np.array([this_set.images[:,c] for c in pixels]))
    return mnist2

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# permuting mnist for 2nd task
mnist2 = permute_mnist_random(mnist, .2)

plt.subplot(1,2,1)
mnist_imshow(mnist.train.images[5])
plt.title("original task image")
plt.subplot(1,2,2)
mnist_imshow(mnist2.train.images[5])
plt.title("new task image");