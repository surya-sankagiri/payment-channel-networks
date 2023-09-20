#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 15:28:25 2021

@author: suryanarayanasankagiri
"""

# This code contains a class and some functions to perform max flow

# input: a directed graph with capacities, a source and a destination
import numpy as np

class network_flow:
    def __init__(self, C, source, destination):
        self.C = C.copy()
        self.size = len(C)
        self.source = source
        self.destination = destination
        self.flow = np.zeros_like(C)
        self.residual = C.copy()
        # do some error handling 

    def shortest_path(self):
        queue = [(None, self.source)]
        unmarked_vertices = np.ones(self.size, dtype = bool)
        parents = np.array([None]*self.size)
        reached = False
        while len(queue) > 0:
            u, v = queue.pop(0)
            if unmarked_vertices[v]:
                parents[v] = u
                unmarked_vertices[v] = False
                if v == self.destination:
                    reached = True
                    break
                for w in np.where(self.residual[v] > 0)[0]:
                    queue.append((v,w))
        if reached:
            parent = parents[self.destination]
            path_capacity = [self.C[parent, self.destination]]
            path = [self.destination]        
            while parent != self.source:
                path.append(parent)
                parent = parents[path[-1]]
                try:
                    path_capacity.append(self.residual[parent, path[-1]])
                except:
                    print(path)
                    print(parents)
                    print(parent)
                    return [], []
            path.append(self.source)
            return np.flip(path), np.flip(path_capacity)
        else:
            return [], []
    
    def push_flow(self):
        path, path_capacity = self.shortest_path()
        if len(path) == 0:
            return [], 0
        else:
            flow_amount = np.min(path_capacity)
            flow_matrix = np.zeros_like(self.C)
            for i in range(len(path)-1):
                flow_matrix[path[i], path[i+1]] = flow_amount
            self.update_flow(flow_matrix)
            return path, flow_amount
        
        
    def update_flow(self, new_flow):
        # print("new flow:\n", new_flow)
        self.flow += new_flow
        # print("current flow:\n", self.flow)
        self.residual -= new_flow
        # print("current flow transpose:\n", np.transpose(new_flow))
        self.residual += np.transpose(new_flow)
        # print("residual:\n", self.residual)
        # print("capacity:\n", self.C)
        # print("residual + flow:\n", self.residual + self.flow - np.transpose(self.flow))
        if not np.array_equal(self.C, self.residual + self.flow - np.transpose(self.flow)):
            print("Something wrong in flow calculations")
        else:
            print("Updated Flow correctly")
            
    def max_flow(self):
        keep_flowing = True
        list_of_paths = []
        list_of_flow_amounts = []
        while keep_flowing:
            path, flow_amount = self.push_flow()
            print("path:", path)
            print("flow amount:", flow_amount)
            list_of_paths.append(path)
            list_of_flow_amounts.append(flow_amount)
            keep_flowing = flow_amount > 0
        return list_of_paths, list_of_flow_amounts


p = 0.3
n = 10
A = np.random.binomial(1,p, (n,n))*np.random.binomial(5, p, (n, n))
for i in range(n):
    A[i, i] = 0
source = 0
destination = 3
print(A)
G = network_flow(A, source, destination)
shortest_path, capacities = G.shortest_path()
print("shortest path:", shortest_path)
list_of_paths, list_of_flow_amounts = G.max_flow()
# print("list of paths:", list_of_paths)
# print("list of flow amounts:", list_of_flow_amounts)
