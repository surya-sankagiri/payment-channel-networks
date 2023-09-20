#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 09:25:46 2021

@author: suryanarayanasankagiri
"""

# this code simulates a simple PCN with three nodes, two channels, 
# a circulation flow and a DAG flow, just like the example given in the
# Spider paper.

# We are going to assign a square-root utility to each flow. 
# Not going to cap it off.
# costs: we are going to have a price that increases linearly with 
# a) capacity usage
# b) imbalance

# the differential equations:

# dx_i/dt = 1/sqrt(x_i+1) - sum of (fraction of capacity used) - sum of (imbalance fraction)
# there are two edges.


# one idea is to just let the differential equations flow
import numpy as np
channel_capacities = 20*np.ones(2) # two channels: 1 <-> 2 and 2 <-> 3
channel_state = channel_capacities/2 # the balance on the lower indexed side

route1 = np.ones(2) # from node 1 to 3
route2 = -1*np.ones(2) # from node 3 to 1
route3 = np.arange(2) # from node 2 to 3
routing_matrix = np.array((route1, route2, route3))

def state_update(channel_state, flows, routing_matrix):
    change = np.dot(flows, routing_matrix)
    return channel_state - change

def calculate_costs(channel_state, routing_matrix):
    channel_costs = channel_capacities - 2*channel_state
    route_costs = np.dot(routing_matrix, channel_costs)
    return route_costs


for t in range(10):
    route_costs = calculate_costs(channel_state, routing_matrix)
    print("route costs:",route_costs)
    flows = (route_costs <= 1).astype(int)
    print("flows:",flows)
    channel_state = state_update(channel_state, flows, routing_matrix)
    print("channel_state:",channel_state)