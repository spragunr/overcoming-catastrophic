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

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

permutation_vector_x, permutation_vector_y = calc_permutation_vectors(mnist.train.images.shape[1], 1, 1)
# permuting mnist for 2nd task
mnist2 = permute_mnist_vector_wrap(mnist, permutation_vector_x, permutation_vector_y)

plt.subplot(1,2,1)
mnist_imshow(mnist.train.images[5])
plt.title("original task image")
plt.subplot(1,2,2)
mnist_imshow(mnist2.train.images[5])
plt.title("new task image");