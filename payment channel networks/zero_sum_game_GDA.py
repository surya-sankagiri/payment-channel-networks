#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 13:09:30 2022

@author: suryanarayanasankagiri
"""

# This code is to check whether the gradient descent ascent algorithm works for zero-sum games.
import numpy as np
import matplotlib.pyplot as plt

def create_random_game(n_x, n_y):
    return np.random.rand(n_x, n_y), np.ones(n_x)/n_x, np.ones(n_y)/n_y
    
def project_onto_prob_space(x):
    n = len(x)
    unit_ones = np.ones(n)/n
    projection = np.dot(x, unit_ones)*unit_ones
    return x - np.sum(x)/n

def gradient_step_with_nonnegativity(x, direction, step_size=0.1):
    v = x + step_size*direction
    a = np.argmin(v)
    b = v[a]
    if b >= 0:
        return v
    else:
        epsilon = x[a]/(x[a]-b)
        return x - epsilon*step_size*direction
    
def one_iteration(A, x, y, epsilon = 0.1, delta = 0.1):
    grad_x = np.dot(A, y)
    grad_x = project_onto_prob_space(grad_x)
    new_x = gradient_step_with_nonnegativity(x, grad_x, epsilon)
    grad_y = np.dot(A, new_x)
    grad_y = project_onto_prob_space(grad_y)
    new_y = gradient_step_with_nonnegativity(y, -grad_y, epsilon)
    new_x = (1-delta)*x + delta*new_x
    return new_x, new_y
    
n_x = 3
n_y = 3
T = 100000
x = np.zeros((T, n_x))
y = np.zeros((T, n_y))
A, x[0], y[0] = create_random_game(n_x, n_y)
for t in range(T-1):
    x[t+1], y[t+1] = one_iteration(A, x[t], y[t], 0.1, 0.1)
    
plt.figure()
plt.plot(np.sum(x, axis=1))
for i in range(n_x):
    plt.plot(x[:,i])

plt.figure()
plt.plot(np.sum(y, axis=1))
for i in range(n_y):
    plt.plot(y[:,i])

