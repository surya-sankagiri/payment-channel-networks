#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 09:31:56 2021

@author: suryanarayanasankagiri
"""

# this code generalizes the code in max_circulation.py
from utils.APSP_class import APSP
import numpy as np
# np.random.seed(0)
import matplotlib.pyplot as plt

def create_demand_matrix(n, mean = 5, sparsity = 0.5):
    D = np.random.binomial(1, sparsity, (n,n))*np.random.poisson(mean, (n,n))
    for i in range(n):
        D[i,i] = 0
    print("Demand Matrix:")
    print(D)
    return D

def create_random_graph(n, k): 
    # creates a k-out random graph
    # and then converts it into an undirected graph
    # self.G is a symmetric, 0-1 matrix
    G = np.zeros((n, n), dtype = bool)
    k = min(k, n-1)
    for i in range(n):
        a = np.concatenate((np.arange(i), np.arange(i+1, n)))        
        G[i, np.random.choice(a, size = k, replace = False)] = True
    G = G + np.transpose(G) #this step makes G undirected
    G = G.astype(int)
    print("Adjacency Matrix:")
    print(G)
    return G
    
def create_capacities(G, mean = 5.0):
    # add weights to the edges
    # self.capacities is a symmetric matrix, with 0 wherever there are no edges
    n = len(G)
    weights = np.random.poisson(mean/2, size = (n, n))
    capacities =G*(weights + np.transpose(weights))
    print("Capacity Matrix:")
    print(capacities)
    return capacities
                

class PCN(APSP):
    def __init__(self, G, C, D, params):
        self.n = len(G)
        self.G = G
        self.capacities = C
        self.demands = D
        self.initialize_distance()
        self.initialize_next()
        self.FloydWarshall()
        self.construct_shortest_paths()
        self.K = params["num_paths"]
        self.construct_random_paths(self.K)
        T = params["T"]
        self.flows = np.zeros((T, self.n, self.n, self.K))
        self.demand_prices = np.zeros((T, self.n, self.n))
        self.edge_capacity_prices = np.zeros((T, self.n, self.n))
        self.edge_imbalance_prices = np.zeros((T, self.n, self.n))
        self.path_prices = np.zeros((T, self.n, self.n, self.K))
        self.flows_per_edge = np.zeros((T, self.n, self.n))
        self.epsilon = params["epsilon"]
        self.delta = params["delta"]
        self.cutoff = params["cutoff"]

    def calculate_path_prices(self, t):
        # for i in range(self.n):
        #     for j in range(self.n):
        #         for k in range(self.K):
        #             self.path_prices[t,i,j,k] = np.sum(self.random_paths[i,j,k]*(self.edge_capacity_prices[t]+self.edge_imbalance_prices[t]))
        edge_prices = self.random_paths*(self.edge_capacity_prices[t]+self.edge_imbalance_prices[t])
        self.path_prices[t] = np.sum(edge_prices, axis=(3,4))
        self.path_prices[t] += np.expand_dims(self.demand_prices[t], -1)
        
    def calculate_flows_per_edge(self, t):
        # for i in range(self.n):
        #     for j in range(self.n):
        #         for k in range(self.K):
        #             self.flows_per_edge[t] += self.flows[t,i,j,k]*self.random_paths[i,j,k]
        self.flows_per_edge[t] = np.sum(np.expand_dims(self.flows[t], axis=(3,4))*self.random_paths,axis=(0,1,2))

    def update_flows_and_prices(self, t):
        self.calculate_path_prices(t)
        self.flows[t+1] = self.flows[t] + self.epsilon*(1-self.path_prices[t])
        self.flows[t+1] = np.maximum(0, self.flows[t+1])

        self.demand_prices[t+1] = self.demand_prices[t] + self.epsilon*(np.sum(self.flows[t+1], axis=-1) - self.demands)
        self.demand_prices[t+1] = np.maximum(0, self.demand_prices[t+1])

        self.calculate_flows_per_edge(t+1)
        self.edge_capacity_prices[t+1] = self.edge_capacity_prices[t] + self.epsilon*(self.flows_per_edge[t+1] + np.transpose(self.flows_per_edge[t+1]) - self.capacities)
        self.edge_capacity_prices[t+1] = np.maximum(0, self.edge_capacity_prices[t+1])
        self.edge_imbalance_prices[t+1] = self.edge_imbalance_prices[t] + self.epsilon*(self.flows_per_edge[t+1] - np.transpose(self.flows_per_edge[t+1]))

        self.flows[t+1] = (1-self.delta)*self.flows[t] + self.delta*self.flows[t+1]

    # def plot_variable_trajectory(self, indices, separate_paths=True):
    #     # complete this. make the price plotting more informative; like for demand price,
    #     # note the actual amount of flow and the demand, and make sure complementary slackness is satisfied
    #     # indices[0] is a list of indices for which 
    #     plt.close("all")
    #     for key, value in indices:
    #         if key == "Flows":
    #             data = self.flows
    #         elif key == "Flows on edges":
    #             data = self.flows_per_edge
    #         elif key == "Demand prices":
                
            
    #     if len(indices[0]) > 0:
    #         plt.figure()
    #         for index in indices:
    #             i = index[0]
    #             j = index[1]
    #             if separate_paths:
    #                 for k in range(self.K):
    #                     plt.plot(self.flows[:,i,j,k], label=str(i)+","+str(j)+","+str(k))
    #             else:
    #                 plt.plot(np.sum(self.flows[:,i,j], axis=-1), label=str(i)+","+str(j))
    #         plt.legend()
    #         plt.title("Flows")
    #         plt.xlabel("Time steps")
        
    #     if len(indices[1]) > 0:
    #         plt.figure()
    #         for index in indices:
    #             i = index[0]
    #             j = index[1]
    #             plt.plot(self.demand_prices[:,i,j], label=str(i)+","+str(j))
    #         plt.legend()
    #         plt.title("Demand prices")
    #         plt.xlabel("Time steps")
        
    #     if len(indices[2]) > 0:
    #         plt.figure()
    #         for index in indices:
    #             i = index[0]
    #             j = index[1]
    #             plt.plot(self.edge_capacity_prices[:,i,j], label=str(i)+","+str(j))
    #         plt.legend()
    #         plt.title("Edge capacity prices")
    #         plt.xlabel("Time steps")

    #     if len(indices[3]) > 0:
    #         plt.figure()
    #         for index in indices:
    #             i = index[0]
    #             j = index[1]
    #             plt.plot(self.edge_imbalance_prices[:,i,j], label=str(i)+","+str(j))
    #         plt.legend()
    #         plt.title("Edge capacity prices")
    #         plt.xlabel("Time steps")

    def compute_steady_state(self):
        self.flow_mean = np.mean(self.flows[-self.cutoff:], axis=0)
        self.demand_prices_mean = np.mean(self.demand_prices[-self.cutoff:], axis=0)
        self.edge_capacity_prices_mean = np.mean(self.edge_capacity_prices[-self.cutoff:], axis=0)
        self.edge_imbalance_prices_mean = np.mean(self.edge_imbalance_prices[-self.cutoff:], axis = 0)
        self.flows_per_edge_mean = np.mean(self.flows_per_edge[-self.cutoff:], axis = 0)
        
    def check_optimality(self, tolerance=0.01):
        primal = np.sum(self.flow_mean)
        dual = np.sum(self.demand_prices_mean*self.demands) + np.sum(self.edge_capacity_prices_mean*self.capacities)/2.0
        if abs(primal-dual)/abs(primal+dual) < tolerance:
            print("Optimal solution reached")
        else:
            print("Solution not optimal")

    def check_feasibility(self, tolerance=0.01):
        if np.sum(self.flow_mean < -tolerance) > 0:
            print("Some flows negative")
        else:
            print("All flows non-negative")

        if np.sum(self.demand_prices_mean < -tolerance) > 0:
            print("Some demand negative")
        else:
            print("All demand prices non-negative")

        if np.sum(self.edge_capacity_prices_mean < -tolerance) > 0:
            print("Some edge capacity prices negative")
        else:
            print("All edge capacity prices non-negative")
            
        A = self.edge_imbalance_prices_mean
        if np.sum(np.abs(A + np.transpose(A)) > tolerance) > 0:
            print("Some edge imbalance prices are not opposites")
        else:
            print("All edge imbalance prices are negatives of each other")

    def check_complimentary_slackness(self, tolerance=0.01):
        slackness_1 = (self.demands - np.sum(self.flow_mean, axis=-1))*self.demand_prices_mean
        if np.sum(np.abs(slackness_1) > tolerance) > 0:
            print("Complimentary slackness of demand prices not satisfied")
        else:
            print("Complimentary slackness of demand prices is satisfied")

        slackness_2 = (self.capacities - 2*self.flows_per_edge_mean)*self.edge_capacity_prices_mean
        if np.sum(np.abs(slackness_2) > tolerance) > 0:
            print("Complimentary slackness of edge capacity prices not satisfied")
        else:
            print("Complimentary slackness of edge capacity prices is satisfied")

if __name__ == "__main__":

    n = 20
    k = 2
    num_paths = 3
    epsilon = 0.1
    delta = 0.1
    T = 50000
    cutoff = int(T/10)
    
    # need to play with the parameters epsilon, delta as a function of n, k, num_paths
    # eps = del = 0.1 seems to be working for a few examples with n = 5, 10.
    D = create_demand_matrix(n)
    G = create_random_graph(n, k)
    C = create_capacities(G)
    params = {"T": T, "num_paths": num_paths, "epsilon": epsilon, "delta":delta, "cutoff": cutoff}
    
    myPCN = PCN(G, C, D, params)
    for t in range(T-1):
        myPCN.update_flows_and_prices(t)
    myPCN.compute_steady_state()
    myPCN.check_feasibility()
    myPCN.check_complimentary_slackness()
    myPCN.check_optimality()
