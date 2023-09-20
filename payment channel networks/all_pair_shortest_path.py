#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:03:32 2021

@author: suryanarayanasankagiri
"""

# This code is used to generate the shortest path among pairs of nodes in a simple (undirected) weighted graph
import numpy as np

# The following class is a collection of functions that computes the 
# shortest path among all pairs of nodes in a simple graph.
# It can also compute some random paths, by routing through an arbitrary node
#using the shortest
# Any path is represented as a 0-1 matrix of size n X n
class APSP():
    def __init__(self, G, random_paths = 0):
        self.G = G
        self.n = len(G)
        self.construct_shortest_paths()
        if random_paths > 0:
            self.construct_random_paths(random_paths)

    def initialize_distance(self):
        self.distance=self.G.copy()
        self.L = np.sum(np.abs(self.G)) # a proxy for infinity
        self.distance[self.distance==0] = self.L # set infinite distance among nodes not connected by an edge
        for v in range(self.n):
            self.distance[v,v] = 0 # set self distance to zero

    def initialize_next(self):
        self.next = [[set() for x in range(self.n)] for y in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if self.distance[i,j] < self.L:
                    self.next[i][j] = {j}

    def FloydWarshall(self):
        self.initialize_distance()
        self.initialize_next()
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if self.distance[i,j] > self.distance[i,k] + self.distance[k,j]:
                        self.distance[i,j] = self.distance[i,k] + self.distance[k,j]
                        self.next[i][j] = self.next[i][k].copy()
                    elif self.distance[i,j] == self.distance[i,k] + self.distance[k,j] and k!= i:
                        self.next[i][j].update(self.next[i][k])
        for i in range(self.n):
            for j in range(self.n):
                self.next[i][j] = list(self.next[i][j])
                print(i,"->",j,":",self.next[i][j])
        # print(self.distance)

    def construct_shortest_paths(self):
        self.FloydWarshall()
        print("All shortest paths")
        self.shortest_paths = dict()
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                self.shortest_paths[(i,j)] = [np.zeros((self.n, self.n))]
                u = i
                path = [u]                    
                while u != j:
                    v = np.random.choice(self.next[u][j])
                    self.shortest_paths[(i,j)][0][u,v] = 1
                    path.append(v)
                    u = v
                print(i,"->",j,":",path)

    # This function generates a random path from i -> j by picking a random node k
    # and merging the shortest path from i -> k and k -> j
    def construct_random_paths(self, K=1):
        self.construct_shortest_paths()
        self.random_paths = dict()
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                excluded = [i, j]
                k = 0
                self.random_paths[(i,j)] = []
                while k < K:
                    u = np.random.randint(0, self.n)
                    if u in excluded:
                        continue
                    excluded.append(u)
                    self.random_paths[(i,j)].append(self.shortest_paths[i,u][0] + self.shortest_paths[u,j][0])
                    k += 1
                
def print_path(start, end, route_matrix):
    path = [start]
    tracker = route_matrix.copy()
    u = start
    while u!= end:
        outgoing_edges = tracker[u]
        v = np.min(np.where(outgoing_edges > 0)[0])
        path.append(v)
        tracker[u,v] -= 1
        u = v
    print(i,j,":",path)
    if np.sum(tracker) > 0:
        print("Cycles remaining")

## To test out the code
if __name__ == "__main__":
    np.random.seed(4)
    n = 10
    K = 1
    A = np.random.binomial(1, 0.4, (n, n))
    for i in range(n):
        A[i,i] = 0
    print(A)
    G = APSP(A, K)
    # G.FloydWarshall()
    # G.construct_shortest_paths()
    # G.construct_random_paths(K)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(K):
                print_path(i, j, G.random_paths[(i,j)][k])

# Things to do:
# 1) extend shortest path to max capacity path
# 2) randomize on the choice of the shortest path or max capacity path, 
# i.e., update with some probability if one finds a path of equal length/capacity
# 3) write a program that always prints the whole path with all cyles
# i.e., extends the functionality of print_path
# 4) write a program that cleans up back-and-forth loops in a path, 
# as that might be quite useless in our context.
