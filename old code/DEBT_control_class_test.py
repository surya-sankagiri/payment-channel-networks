#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:17:17 2022

@author: suryanarayanasankagiri
"""

# This python file contains the code to regulate flows based on prices
# The system proceeds in discrete-time. In each step:
# The channels in the network quote prices to the edges
# The controller decides whether or not to push flow through
# The network attempts to route the money

import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from utils.PCN_Dynamics_class import PCNDynamics
from utils.simple_functions import *

def create_demand_matrix(demand_params, n):
    mean = demand_params["mean"]
    sparsity = demand_params["sparsity"]
    demands = np.random.binomial(1, sparsity, (n, n))*np.random.poisson(mean, (n, n))
    for i in range(n):
        demands[i,i] = 0
    print("Demand Matrix:")
    print(demands)
    demand_params["demand_matrix"] = demands
    return demand_params

class DEBT_Control_Protocol(PCNDynamics):
    # This function initializes the whole system
    def __init__(self, params, demand_params, price_params):
        super().__init__(params) # sets up the PCN and establishes its routes
        self.demand_distribution = demand_params["distribution"] # whether demands are random (IID) or constant
        self.demands = demand_params["demand_matrix"] # creates the matrix of source-destination demand rates
        self.channel_prices = np.zeros((self.n, self.n))
        self.price_threshold = dict()
        self.price_sensitivity = dict()
        for i in range(self.n):
            for j in range(self.n):
                self.price_threshold[(i,j)] = price_params["threshold"][i,j]
                self.price_sensitivity[(i,j)] = price_params["sensitivity"][i,j]
        self.price_stepsize = price_params["stepsize"]



    def compute_flows(self):
        # the following three functions generate quantities which are dictionaries with node pairs as keys
        self._generate_demands() # generates instantaneous demands from the demand matrix
        self._generate_path_prices() # generates path prices from channel prices
        self._decide_flows() # decides the flow in each path as a function of the demand and the path prices

    def _generate_demands(self):
        self.current_demand_matrix = np.zeros((self.n, self.n))
        if self.demand_distribution == "deterministic":
            self.current_demand_matrix = self.demands.copy()
        elif self.demand_distribution == "Poisson":
            self.current_demand_matrix = np.random.poisson(self.demands)
        self.current_demand = dict()
        for i in range(self.n):
            for j in range(self.n):
                if self.current_demand_matrix[i,j] > 0:
                    self.current_demand[(i,j)] = self.current_demand_matrix[i,j]

    def _generate_path_prices(self):
        self.path_prices = dict()
        for node_pair in self.paths.keys():
            self.path_prices[node_pair] = [np.sum(path*self.channel_prices) for path in self.paths[node_pair]]
        # for i in range(self.n):
        #     for j in range(self.n):
        #         if (i,j) not in self.paths.keys():
        #             continue
        #         self.path_prices[(i,j)] = [np.sum(path*self.link_prices) for path in self.paths[(i,j)]]

    def _decide_flows(self):
        self.flow_requests = dict()
        for node_pair in self.paths.keys():
            self.flow_requests[node_pair] = [0]*len(self.paths[node_pair])
            max_price = self.price_threshold[node_pair]
            try:
                demand = self.current_demand[node_pair]
            except KeyError:
                continue # demand must be zero so the requested flow must be zero
            self.flow_requests[node_pair] = self._flow_calculator(max_price, demand, self.path_prices[node_pair])

        # for i in range(self.n):
        #     for j in range(self.n):
        #         if (i,j) not in self.paths.keys():
        #             continue
        #         self.flow_requests[(i,j)] = [0]*len(self.path_prices[(i,j)])
        #         min_price = np.min(self.path_prices[(i,j)])
        #         if min_price < self.price_threshold:
        #             self.flow_requests[(i,j)][np.argmin(self.path_prices[(i,j)])] = self.current_demands[i,j]

    # def decide_flows_softmin(self, c):
    #     self.flow_requests = dict()
    #     for i in range(self.n):
    #         for j in range(self.n):
    #             if (i,j) not in self.paths.keys():
    #                 continue
    #             weights = softmax(self.path_prices[(i,j)], c)
    #             min_price = np.min(self.path_prices[(i,j)])
    #             flow_amount = self.flow_calculator(min_price, self.current_demands[i,j])
    #             self.flow_requests[(i,j)] = list(weights*flow_amount)
    #             # print(i,j)
    #             # print(self.path_prices[(i,j)])
    #             # print(self.flow_requests[(i,j)])

    # def _flow_calculator(self, price, max_amount):
    #     return max_amount
        # if price < self.price_threshold_low:
        #     return max_amount
        # elif price < self.price_threshold_hi:
        #     return max_amount*(self.price_threshold_hi - price)/(self.price_threshold_hi - self.price_threshold_low)
        # else:
        #     return 0.0
        
    def loop(self):
        self.single_step()
        self.flows_per_edge, self.channels_reset = self.execute(self.flow_requests)
        # print("\n Time slot:",t)
        # for key, value in self.flow_requests.items():
        #     print("Source:", key[0], "Destination:", key[1], "Flows requested", value, "Flow routed", routed[key])
        # print("Balances:\n", self.balances)
        # print("Link prices:\n", self.link_prices)

if __name__ == "__main__":
    params = dict() # a dictionary of parameters that creates the PCN and its paths
    params["custom"] = False
    # params["capacity_matrix"] = np.array([[0, 100, 100], [100, 0, 100], [100, 100, 0]])
    # params["capacity_matrix"] = np.array([[0, 100, 0], [100, 0, 100], [0, 100, 0]])
    params["num_vertices"] = 6
    params["average_degree"] = 3
    params["average_capacity"] = 200
    params["custom_routes"] = False
    params["num_paths"] = 2
    params["price_update_step_size"] = 0.005
    if params["custom"]:
        n = len(params["capacity_matrix"])
    else:
        n = params["num_vertices"]
    
    demand_params = dict()
    # demand_params["demand_matrix"] = np.array([[0, 0, 10], [10, 0, 10], [10, 0, 0]])
    demand_params["mean"] = 10
    demand_params["sparsity"] = 0.5
    demand_params["distribution"] = "deterministic" # or "Poisson"
    demand_params = create_demand_matrix(demand_params, n)
    price_threshold = 3.0*np.ones((n,n)) # every (i,j) pair can have a distinct price thresold

    T = 3000
    averaging_window = 1

    myFC = DEBT_Control_Protocol(params, demand_params, price_threshold)
    # create arrays to store flows and prices
    flows_data = dict()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            flows_data[(i,j)] = np.zeros((T,len(myFC.paths[(i,j)])))
    path_price_data = dict()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            path_price_data[(i,j)] = np.zeros((T,len(myFC.paths[(i,j)])))
    edge_price_data = np.zeros((T, n, n))
    reset_data = np.zeros((T, n, n))


    for t in range(T):
        myFC.loop()
        edge_price_data[t] = myFC.link_prices.copy()
        for key in flows_data.keys():
            flows_data[key][t] = myFC.flow_requests[key]
            path_price_data[key][t] = myFC.path_prices[key]
            reset_data[t] = myFC.channels_reset

    alphabets = ['A', 'B', 'C', 'D', 'E', 'F']
    plt.close("all")
    plt.figure(figsize=(6,4))
    for i in range(n):
        for j in range(n):
            if myFC.demands[i,j] > 0:
                plt.plot(moving_average(np.sum(flows_data[(i,j)], axis=1), averaging_window), label = alphabets[i] + "->" + alphabets[j])
                # plt.plot(total_average(np.sum(flows_data[(i,j)], axis=1)), label = str(i) + "->" + str(j) + "flow; demand "+str(myFC.demands[i,j]))
                # plt.plot(np.sum(flows_data[(i,j)], axis=1), label = alphabets[i] + "->" + alphabets[j])
    # plt.plot(flows_data[(0,1)][:,0], label = "short path", marker = 's', linestyle='none')
    # plt.plot(flows_data[(0,1)][:,1], label = "long path", marker = 's', linestyle='none')
    # plt.plot(np.sum(flows_data[(0,1)], axis=1), label = "total flow")
    # plt.plot(np.sum(flows_data[(1,0)], axis=1), label = "B->A flow", marker = 's', linestyle='none')
    # plt.plot(np.sum(flows_data[(1,2)], axis=1), label = "B->C flow", marker = 'v', linestyle='none',ms=4)
    # plt.plot(np.sum(flows_data[(0,2)], axis=1), label = "A->C flow", marker = 's', linestyle='none')
    # plt.plot(np.sum(flows_data[(2,0)], axis=1), label = "C->A flow", marker = 'v', linestyle='none',ms=4)
    # plt.legend()
    plt.xlabel("Time", size = 14)
    plt.xticks(fontsize=10)
    plt.ylabel("Flow Amount", size = 14)
    plt.yticks(fontsize=10)
    plt.title("Flows as a function of time", size=16)
    plt.tight_layout()
    # plt.savefig('flows_1.pdf')

    plt.figure(figsize=(6,4))
    for i in range(n):
        for j in range(n):
            if myFC.demands[i,j] > 0:
                plt.plot(moving_average(edge_price_data[:, i, j], averaging_window), label = alphabets[i] + "->" + alphabets[j])
                # plt.plot(total_average(np.min(path_price_data[(i,j)], axis=1)), label = str(i) + "->" + str(j) + " price")
                # plt.plot(np.min(path_price_data[(i,j)], axis=1), label = alphabets[i] + "->" + alphabets[j])
    # plt.plot(path_price_data[(0,1)][:,0], label = "short path")
    # plt.plot(path_price_data[(0,1)][:,1], label = "long path")
    # plt.plot(np.min(path_price_data[(0,1)], axis=1), label = "minimum price", marker = 's', linestyle = 'none')
    # plt.plot(np.min(path_price_data[(1,0)], axis=1), label = "B->A price", marker = 's', linestyle='none')
    # plt.plot(np.min(path_price_data[(1,2)], axis=1), label = "B->C price", marker = 'v', linestyle='none',ms=4)
    # plt.plot(np.min(path_price_data[(0,2)], axis=1), label = "A->C price", marker = 's', linestyle='none')
    # plt.plot(np.min(path_price_data[(2,0)], axis=1), label = "C->A price", marker = 'v', linestyle='none',ms=4)

    plt.axhline(y=price_threshold[0], color='k', linestyle='--', label= "price threshold")
    plt.axhline(y=price_threshold[1], color='k', linestyle='--')
    # plt.legend()
    plt.xlabel("Time", size = 14)
    plt.xticks(fontsize=10)
    plt.ylabel("Price", size = 14)
    plt.yticks(fontsize=10)
    plt.title("Path prices as a function of time", size = 16)
    plt.tight_layout()
    # plt.savefig('prices_1.pdf')

    plt.figure(figsize=(6,4))
    for i in range(n):
        for j in range(i):
            if myFC.capacities[i, j] > 0:
                plt.plot(moving_average(np.min(path_price_data[(i,j)], axis=1), 1), label = "channel " + alphabets[i] + "-" + alphabets[j])
                # plt.plot(total_average(np.min(path_price_data[(i,j)], axis=1)), label = str(i) + "->" + str(j) + " price")
                # plt.plot(np.min(path_price_data[(i,j)], axis=1), label = alphabets[i] + "->" + alphabets[j])
    # plt.plot(path_price_data[(0,1)][:,0], label = "short path")
    # plt.plot(path_price_data[(0,1)][:,1], label = "long path")
    # plt.plot(np.min(path_price_data[(0,1)], axis=1), label = "minimum price", marker = 's', linestyle = 'none')
    # plt.plot(np.min(path_price_data[(1,0)], axis=1), label = "B->A price", marker = 's', linestyle='none')
    # plt.plot(np.min(path_price_data[(1,2)], axis=1), label = "B->C price", marker = 'v', linestyle='none',ms=4)
    # plt.plot(np.min(path_price_data[(0,2)], axis=1), label = "A->C price", marker = 's', linestyle='none')
    # plt.plot(np.min(path_price_data[(2,0)], axis=1), label = "C->A price", marker = 'v', linestyle='none',ms=4)

    # plt.axhline(y=price_threshold[0], color='k', linestyle='--', label= "price threshold")
    # plt.axhline(y=price_threshold[1], color='k', linestyle='--')
    plt.legend()
    plt.xlabel("Time", size = 14)
    plt.xticks(fontsize=10)
    plt.ylabel("Price", size = 14)
    plt.yticks(fontsize=10)
    plt.title("Edge prices as a function of time", size = 16)
    plt.tight_layout()

    plt.figure(figsize=(6,4))
    for i in range(n):
        for j in range(i):
            if myFC.capacities[i, j] > 0:
                plt.plot(reset_data[:,i,j], label = "channel " + alphabets[i] + "-" + alphabets[j], marker = 's', linestyle='none')
    plt.legend()
    plt.xlabel("Time", size = 14)
    plt.xticks(fontsize=10)
    plt.title("Channel Resets", size = 16)
    plt.tight_layout()


    # plt.figure()
    # for t in range(T):
    #     if reset_data[t, 1, 0]:
    #         plt.axvline(x = t, color = 'b')
    #     if reset_data[t, 1, 2]:
    #         plt.axvline(x = t, color = 'r')
    # plt.xlim([0, T])

    # for i in range(n):
    #     for j in range(n):
    #         if myFC.demands[i,j] > 0:
    #             plt.figure()
    #             plt.plot(moving_average(np.sum(flows_data[(i,j)], axis=1), averaging_window), label = str(i) + "->" + str(j) + "flow; demand "+str(myFC.demands[i,j]))
    #             plt.plot(moving_average(np.min(path_price_data[(i,j)], axis=1), averaging_window), label = str(i) + "->" + str(j) + " price") #plot on other y-axis
    # plt.legend()
    # what quantities should be plotted?
    # the flows, among every pair there is a demand
    # the path price, among every pair there is a demand
    # time averaged versions of these.

    ## SUMMARY OF SIMULATIONS ON March 3, 2022
    # When the demand is [[0, 0, 10], [10, 0, 10], [10, 0, 0]], i.e., classical deadlock example,
    # Then any (small enough) strictly positive threshold will give the optimal flow.
    # However, if the threshold is too large, the deadlock-causing flow cannot be prevented,
    # The steady-state flow will become zero

    # When the demand is [[0, 4, 0], [3, 0, 2], [5, 0, 0]], i.e., the non-trivial max-circulation example,
    # Then, if the threshold is too small, the steady-state flow is sub-optimal.
    # However, if the threshold is too large, even then, the steady-state flow is optimal.
    # This is because the demands are deadlock-free.
    # We do run into the issue that flows are requested, but cannot be routed.


    ## SUGGESTION: TRY A PRIMAL ALGORITHM OR DUAL ALGORITHM.
    ## INVOKE NEDIC'S RESULT

