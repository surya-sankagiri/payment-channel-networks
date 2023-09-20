#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 07:06:30 2021

@author: suryanarayanasankagiri
"""

## This code implements a primal-dual algorithm for the max circulation 
## optimization problem, which is a way to model the demands-alone problem.
## We solve the formulation with flows along edges, 
## which leads to two kinds of Lagrange multipliers.

import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_demand_matrix(n, mean = 5, sparsity = 0.5):
    D = np.random.binomial(1, sparsity, (n,n))*np.random.exponential(mean, (n,n))
    for i in range(n):
        D[i,i] = 0
    return D

def convert_matrix(lamda):
    n = len(lamda)
    return np.reshape(np.tile(lamda, n), (n,n))

def take_average(x, t, delta):
    return (1-delta)*x[t] + delta*x[t+1]

def constant_out(demands, x, c):
    n = len(demands)
    z = x.copy()
    for i in range(n):
        for j in range(n):
            if demands[i,j] == 0:
                z[i,j] = c
    return z

def Lagrangian(x, l, m, D):
    S = np.sum(m*D)
    L = convert_matrix(l)
    M = 1 - (m + L.transpose() - L)
    S += np.sum(x*M)
    return S

def check_optimality(x, y, d, cutoff=1000, tolerance=0.01):
    x_mean = np.mean(x[-cutoff:], axis=0)
    y_mean = np.mean(y[-cutoff:], axis=0)
    primal = np.sum(x_mean)
    dual = np.sum(y_mean*d)
    return abs(primal-dual)/abs(primal+dual) < tolerance

def check_convergence(x, cutoff=1000, tolerance=0.1):
    x_max = np.max(x[-cutoff:], axis=0)
    x_min = np.min(x[-cutoff:], axis=0)
    return np.max(x_max-x_min) < tolerance

def execute(demands, n, T, eps_primal, eps_dual, delta):
    flows = np.zeros((T,n,n))
    mu = np.ones((T,n,n))
    lamda = np.zeros((T,n))
    lamda_matrix = np.zeros((T,n,n))
    lagrangian = np.zeros(T)
    
    for t in range(T-1):
        lagrangian[t] = Lagrangian(flows[t], lamda[t], mu[t], demands)
    
        lamda_matrix[t] = convert_matrix(lamda[t])
        flows[t+1] = flows[t] + eps_primal[t]*(1 - (mu[t] + lamda_matrix[t].transpose() - lamda_matrix[t]))
        flows[t+1] = np.maximum(flows[t+1], 0)
        # flows[t+1] = constant_out(demands, flows[t+1], 0)

        mu[t+1] = mu[t] + eps_dual[t]*(flows[t+1] - demands)
        mu[t+1] = np.maximum(mu[t+1], 0)
        # mu[t+1] = constant_out(demands, mu[t+1], 1)
    
        lamda[t+1] = lamda[t] + eps_dual[t]*(np.sum(flows[t+1], axis = 1) - np.sum(flows[t+1], axis = 0))
        
        # flows[t+1] = take_average(flows, t, delta) # not setting delta here makes it oscillate
        # mu[t+1] = take_average(mu, t, delta) # setting delta here takes more time to converge
        # lamda[t+1] = take_average(lamda, t, delta) # setting delta here takes more time to converge

        lamda_matrix[t+1] = convert_matrix(lamda[t+1])
        flows[t+1] = flows[t] + eps_primal[t]*(1 - (mu[t+1] + lamda_matrix[t+1].transpose() - lamda_matrix[t+1]))
        flows[t+1] = np.maximum(flows[t+1], 0)
        
    lagrangian[-1] = Lagrangian(flows[-1], lamda[-1], mu[-1], demands)
    return flows, mu, lamda, lagrangian

def plot_variable_trajectory(demands, flows, mu, lamda):
    plt.close("all")
    plt.figure()
    for i in range(n):
        for j in range(n):
            if demands[i,j] > 0:
                plt.plot(flows[:,i,j], label=str(i)+","+str(j))
    # plt.legend()
    plt.title("Flows")
    plt.xlabel("Time steps")
        
    plt.figure()
    for i in range(n):
        for j in range(n):
            if demands[i,j] > 0:
                plt.plot(mu[:,i,j], label=str(i)+","+str(j))
    # plt.legend()
    plt.title("Edge prices")
    plt.xlabel("Time steps")
    
    plt.figure()
    for i in range(n):
        plt.plot(lamda[:,i], label=str(i))
    # plt.legend()
    plt.title("Node potentials")
    plt.xlabel("Time steps")
    
def plot_squared_error(flows, mu, lamda):
    x = np.sum(np.sum((flows - flows[-1])**2, axis=1), axis = 1)
    y = np.sum(np.sum((mu - mu[-1])**2, axis=1), axis = 1)
    z = np.sum((lamda - lamda[-1])**2, axis=1)
    plt.figure()
    plt.plot(x+y+z)

def plot_asymptotics(x, cutoff = 2000, index=None):
# This function is to plot the rate of convergence of the envelope for individual flows and prices
    plt.figure()
    for i in range(n):
        for j in range(n):
            if x[-1, i,j] > 0.01:
                z = x[cutoff:,i,j]
                z = np.abs(z - z[-1])
                ar1 = np.where(z[:-1] > z[1:])[0]
                ar2 = np.where(z[:-1] < z[1:])[0]
                i = np.intersect1d(ar1, ar2+1)
                plt.plot(i, z[i], label=str(i) + "," + str(j))
    plt.legend()

def transition_matrices(n, epsilon=0.1, delta=0.1):
    A = np.identity(2*n*n + n)
    A[:n*n, n*n:2*n*n] = -epsilon*np.identity(n*n)
    A[n*n:2*n*n, :n*n] = epsilon*np.identity(n*n)
    M = np.zeros((n, n*n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if k == i:
                    M[k, n*i + j] -= 1
                if k == j:
                    M[k, n*i + j] += 1
    A[:n*n, 2*n*n:] = epsilon*M.transpose()
    A[2*n*n:, :n*n] = -epsilon*M
    
    B = A.copy()
    B[n*n:2*n*n, n*n:2*n*n] -= epsilon*epsilon*np.identity(n*n)
    B[2*n*n:,2*n*n:] -= epsilon*epsilon*np.dot(M,M.transpose())
    B[n*n:2*n*n, 2*n*n:] += epsilon*epsilon*M.transpose()
    B[2*n*n:, n*n:2*n*n] -= epsilon*epsilon*M

    C = B.copy()
    C[:n*n, n*n:] *= delta
    
    return A, B, C, M

def max_eigenvalues(epsilon,delta,n=40,num_eigs=1):
    A,B,C,M = transition_matrices(n, epsilon, delta)
    w,v = np.linalg.eig(C)
    x = np.abs(w)
    max_eigs = np.zeros(num_eigs)
    for i in range(num_eigs):
        max_eigs[i] = max(x)
        x = np.delete(x, np.argmax(x))
    return max_eigs

def plotmaxeig(num_eigs=1):
    plt.close("all")
    eps = np.arange(0.01, 0.25, 0.01)
    delt = np.arange(0.01, 0.25, 0.01)
    E, D = np.meshgrid(eps, delt)
    s = E.shape
    s += (num_eigs,)
    max_eigs = np.zeros(s)
    for i in range(len(eps)):
        for j in range(len(delt)):
            max_eigs[i,j] = max_eigenvalues(eps[i],delt[j],num_eigs=num_eigs)
    
    for n in range(num_eigs):
        Z = max_eigs[:,:,n]
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(E, D, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
        ax.set_xlabel('epsilon')
        ax.set_ylabel('delta')
        ax.set_zlabel('max_eigenvalue')
        # ax.set_zlim([0.8, 1.1])
        ax.set_zlim([0, int(np.max(Z)) + 1])
        print("min value:", np.min(Z))

# write the above two functions a little better, and see how the eigenvalues 
# vary with epsilon, delta and n. Also try to explain why we might have an eigevalue of 1
# plotmaxeig(2)

n = 50
T = 50000
eps_primal = 0.1*np.ones(T)
eps_dual = 0.1*np.ones(T)
delta = 0.1
num_iterations = 1

for k in range(num_iterations):
    demands = create_demand_matrix(n)
    flows, mu, lamda, lagrangian = execute(demands, n, T, eps_primal, eps_dual, delta)
    if check_convergence(flows) and check_convergence(mu) and check_convergence(lamda):
        print("Converged")
    else:
        print("Didn't converge")
        # plot_variable_trajectory(demands, flows, mu, lamda)
    if check_optimality(flows, mu, demands):
        print("Optimal reached")
    else:
        print("Optimal failed")
    plot_squared_error(flows, mu, lamda)
    plot_variable_trajectory(demands, flows, mu, lamda)