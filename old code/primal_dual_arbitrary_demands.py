#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 15:38:30 2022

@author: suryanarayanasankagiri
"""

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
plt.close("all")

def create_demand_matrix(n, mean = 4, sparsity = 0.4):
    D = np.random.binomial(1, sparsity, (n,n))*np.random.poisson(mean, (n,n))
    for i in range(n):
        D[i,i] = 0
    return D

def create_channel_matrix(n, mean = 4, sparsity = 0.4):
    E = np.random.binomial(1, sparsity, (n,n))*np.random.poisson(mean, (n,n))
    for i in range(n):
        E[i,i] = 0
        for j in range(i):
            E[i,j] = E[j,i]
    return E

# suppose x is a vector
# better group it into groups of source-destination pairs.

