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
import logging
from utils.PCN_Dynamics_class import PCNDynamics
from utils.simple_functions import hard_routing, soft_routing

class DEBT_Control_Protocol(PCNDynamics):
    # This function initializes the whole system
    def __init__(self, params, demand_params, price_params):
        logging.info("initializing PCN")
        super().__init__(params) # sets up the PCN and establishes its routes
        logging.info("initializing demands")
        self._initialize_demands(demand_params)
        logging.info("initializing prices")
        self._initialize_prices(price_params)

    def _initialize_demands(self, demand_params):
        self.demand_distribution = demand_params["distribution"] # whether demands are random (IID) or constant
        if demand_params["custom demands"] == True:
            logging.info("custom demand matrix")
            self.demands = demand_params["demand_matrix"] # the matrix of source-destination demand rates
        else:
            logging.info("random demand matrix")
            self.demands = self._random_demand_matrix(demand_params)
        logging.info("finished initializing demands")

    def _random_demand_matrix(self, demand_params):
        mean = demand_params["mean"]
        sparsity = demand_params["sparsity"]
        demands = np.random.binomial(1, sparsity, (self.n, self.n))*np.random.poisson(mean, (self.n, self.n))
        for i in range(self.n):
            demands[i,i] = 0
        logging.debug("Demand Matrix:")
        logging.debug(demands)
        return demands

    def _initialize_prices(self, price_params):
        self.channel_prices = np.zeros((self.n, self.n))
        self.price_threshold = dict()
        self.price_sensitivity = dict()
        for i in range(self.n):
            for j in range(self.n):
                self.price_threshold[(i,j)] = price_params["threshold"][i,j]
                self.price_sensitivity[(i,j)] = price_params["sensitivity"][i,j]
        self.price_stepsize = price_params["stepsize"]
        logging.info("finished initializing prices")

    def single_step(self, t):
        # the following three functions generate quantities which are dictionaries with node pairs as keys
        logging.info("generating transaction requests")
        self._generate_txn_requests() # generates instantaneous demands from the demand matrix
        logging.info("computing path prices")
        self._generate_path_prices() # generates path prices from channel prices
        logging.info("deciding flows")
        self._decide_flows() # decides the flow in each path as a function of the demand and the path prices
        logging.info("executing flow requests")
        self.execute(self.flow_requests) # defined in base class; executes the flows and updates the balances
        logging.info("updating prices")
        self.update_prices() # updates the prices
        logging.info("finished step %d of protocol", t)

    def _generate_txn_requests(self):
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
        logging.info("finished generating transaction requests")

    def _generate_path_prices(self):
        self.path_prices = dict()
        for node_pair in self.paths.keys():
            self.path_prices[node_pair] = [np.sum(path*self.channel_prices) for path in self.paths[node_pair]]
        logging.info("finished computing path prices")

    def _decide_flows(self):
        self.flow_requests = dict()
        for node_pair in self.paths.keys():
            try:
                demand = self.current_demand[node_pair]
            except KeyError:
                continue # demand must be zero so the requested flow must be zero
            max_price = self.price_threshold[node_pair]
            logging.info("calculating flows for %s node pair", node_pair)
            self._flow_calculator(max_price, demand, node_pair)

    def _flow_calculator(self, max_price, demand, node_pair):
        if self.price_sensitivity[node_pair] < 0.0001: # equivalent to hard routing
            logging.info("performing hard routing")
            self.flow_requests[node_pair] = hard_routing(self.path_prices[node_pair], max_price, demand)
        else:
            logging.info("performing soft routing")
            self.flow_requests[node_pair] = soft_routing(self.path_prices[node_pair], max_price, demand, self.price_sensitivity[node_pair])
        logging.info("finished calculating flows for %s node pair", node_pair)

    def update_prices(self):
        self.channel_prices += self.price_stepsize*(self.flows_per_edge - self.flows_per_edge.transpose())
        logging.info("finished updating prices")