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
import logging

from utils.PCN_class import PCN

class PCNDynamics(PCN):
    def __init__(self, params):
        super().__init__(params)
        self.balances = self.capacities/2 # initialize the system to have perfectly balanced channels
        logging.debug("Initially, the balances are as follows:")
        logging.debug(self.balances)
        self._reset()
        logging.info("finished initializing PCN")

    def _reset(self):
        self.flows_per_edge = np.zeros((self.n, self.n)) # this is the amount that actually flows in either direction in the previous slot
        self.channels_reset = np.zeros((self.n, self.n), dtype = bool) #indicator of which channels were reset; symmetric matrix

    # The PCN receives flow_requests as a dictionary, where the keys are (source, destination) pairs
    # And the values are a vector of flow amounts, each component for each path.
    # If the flow is a multipath one, more than one flow component will be positive
    def execute(self, flow_requests):
        self._reset()
        self._flows_paths_to_edges(flow_requests)
        self._update_balances()
        return self.flows_per_edge, self.channels_reset

    def _flows_paths_to_edges(self, flow_requests):
        for node_pair, flow_vec in flow_requests.items():
            for p in range(len(flow_vec)):
                self.flows_per_edge += flow_vec[p] * self.paths[node_pair][p]
            logging.info("finished calculating flows for node pair %s", node_pair)

    def _update_balances(self):
        logging.debug("before routing, the balances are as follows:")
        logging.debug(self.balances)
        self.channels_reset = self.balances - self.flows_per_edge < 0
        self.channels_reset += self.balances + self.flows_per_edge.transpose() > self.capacities
        self.balances = (self.capacities/2)*self.channels_reset + self.balances*(1 - self.channels_reset)
        logging.info("finished performing channel resets")
        logging.debug("the following channels were reset:")
        logging.debug(self.channels_reset)

        logging.debug("now the balances are as follows:")
        logging.debug(self.balances)

        self.balances -= self.flows_per_edge
        self.balances += self.flows_per_edge.transpose()
        logging.info("finished updating balances")
        logging.debug("now the balances are as follows:")
        logging.debug(self.balances)
