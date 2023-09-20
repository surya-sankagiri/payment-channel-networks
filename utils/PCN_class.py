#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:57:20 2022

@author: suryanarayanasankagiri
"""

# This python file contains the code to initialize a payment channel network (PCN)
# The class PCN contains information about the topology and the capacities of the graph
# It also contains the set of possible routes between all pairs of nodes
import numpy as np
from utils.APSP_class import APSP
class PCN(APSP):
    # The main function of this class
    def __init__(self, params):
        if params["custom"] == True:
            self.capacities = params["capacity_matrix"]
            self.n = len(self.capacities)
            self.G = (self.capacities > 0).astype(int)
        else:
            self.n = int(params["num_vertices"])
            k = int(params["average_degree"]/2)
            k = min(k, self.n-1)
            self.add_random_edges(k)
            c = params["average_capacity"]
            self.add_random_capacities(c)
        if params["custom_routes"] == True:
            self.paths = params["routes"]
        else:
            self.construct_random_paths(params["num_paths"]-1) #includes construction of shortest paths
            self.paths = self.shortest_paths.copy()
            for i in range(self.n):
                for j in range(self.n):
                    if i == j:
                        continue
                    self.paths[(i,j)] += self.random_paths[(i,j)]

    def add_random_edges(self, k): 
        # creates a k-out random graph and then converts it into an undirected graph
        # self.G is a symmetric, 0-1 matrix
        self.G = np.zeros((self.n, self.n), dtype = bool)
        for i in range(self.n):
            a = np.concatenate((np.arange(i), np.arange(i+1, self.n)))        
            self.G[i, np.random.choice(a, size = k, replace = False)] = True
        self.G = self.G + np.transpose(self.G) #this step makes G undirected
        self.G = self.G.astype(int)
        print("Adjacency Matrix:")
        print(self.G)

    def add_random_capacities(self, c):
        # add weights to the edges
        # self.capacities is a symmetric matrix, with 0 wherever there are no edges
        weights = np.random.poisson(c/2, size = (self.n, self.n))
        self.capacities = self.G*(weights + np.transpose(weights))
        print("Capacity Matrix:")
        print(self.capacities)
