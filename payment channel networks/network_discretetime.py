#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 09:57:53 2022

@author: suryanarayanasankagiri
"""

# This python file contains the code for how a PCN routes flows 
# that are requested, and updates its price based on the flows requested
# The system evolves in discrete-time. It performs the following functions, in the given order:
# 1) The network receives flow requests from nodes, and routes as many of them as possible
#    Those that cannot be served are simply dropped
# 2) Channels rebalance themselves if they are not able to serve flow requests in the current iteration
# 3) The channels update their price based on the flow requests received


import numpy as np
import random

from basicPCN import PCN

class PCNDynamics(PCN):
    # initialize the system to have perfectly balanced channels and the price to be zero
    def __init__(self, params):
        super().__init__(params)
        self.balances = self.capacities/2
        self.flows_per_edge = np.zeros((self.n, self.n)) # this is the amount that actually flows in either direction in the previous slot
        self.channels_reset = np.zeros((self.n, self.n), dtype = bool) #indicator of which channels were reset; symmetric matrix
        self.link_prices = np.zeros((self.n, self.n)) # this is an anti symmetric matrix
        self.requests_per_edge = np.zeros((self.n, self.n)) # this is the amount requested in the previous slot
        self.step_size = params["price_update_step_size"]
    # The PCN receives flow_requests as a dictionary, where the keys are (source, destination)
    # And the values are a vector of flow amounts, each component for each path.
    # If the flow is a multipath one, more than one flow components will be positive
    def execute(self, flow_requests):
        self.requests_per_edge[:,:] = 0
        self.flows_per_edge[:,:] = 0
        self.channels_reset[:,:] = False # stores which channels need to be rebalanced
        routed = dict() # indicator for which flows are routed
        # First, order all flow requests in some random permutation
        flow_requests = list(flow_requests.items())
        random.shuffle(flow_requests)
        for SD_pair, flow_vec in flow_requests:
            # one by one, check whether it is feasible to route 
            # if feasible, add to flows (to be routed)
            # if not, mark appropriate channel for rebalancing
            # in either case, store the request for each channel as potential queue length
            routed[SD_pair] = self.process_flow(SD_pair, flow_vec)
        self.update_balances() # based on self.flows_per_edge, self.channels_reset
        self.update_prices() # based on self.requests_per_edge
        return routed
        
    def process_flow(self, SD_pair, flow_vec):
        flow_per_path = np.zeros((self.n, self.n))
        for n in range(len(flow_vec)):
            flow_per_path += flow_vec[n] * self.paths[SD_pair][n]
        self.requests_per_edge += flow_per_path
        link_saturations = (self.flows_per_edge + flow_per_path) > self.balances
        self.channels_reset += link_saturations
        feasible = np.sum(link_saturations) == 0
        if feasible:
            self.flows_per_edge += flow_per_path
        return feasible
        
    def update_balances(self):
        self.balances -= self.flows_per_edge - self.flows_per_edge.transpose()
        self.channels_reset += self.channels_reset.transpose() # to make it 
        self.balances = self.balances*(1-self.channels_reset) + self.channels_reset*(self.capacities/2)
        
    def update_prices(self):
        self.link_prices += self.step_size*(self.requests_per_edge - self.requests_per_edge.transpose())