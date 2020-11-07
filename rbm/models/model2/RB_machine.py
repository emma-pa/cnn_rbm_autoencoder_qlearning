#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

patterns = -np.ones((14,3,3))

patterns[1] = np.array([[1,-1,-1],[1,-1,-1],[1,-1,-1]])
patterns[2] = np.array([[-1,1,-1],[-1,1,-1],[-1,1,-1]])
patterns[3] = np.array([[-1,-1,1],[-1,-1,1],[-1,-1,1]])
patterns[4] = np.array([[1,1,-1],[1,1,-1],[1,1,-1]])
patterns[5] = np.array([[-1,1,1],[-1,1,1],[-1,1,1]])
patterns[6] = np.array([[1,-1,1],[1,-1,1],[1,-1,1]])
patterns[7] = np.array([[1,1,1],[1,1,1],[1,1,1]])
patterns[8] = np.array([[1,1,1],[-1,-1,-1],[-1,-1,-1]])
patterns[9] = np.array([[-1,-1,-1],[1,1,1],[-1,-1,-1]])
patterns[10] = np.array([[-1,-1,-1],[-1,-1,-1],[1,1,1]])
patterns[11] = np.array([[1,1,1],[1,1,1],[-1,-1,-1]])
patterns[12] = np.array([[-1,-1,-1],[1,1,1],[1,1,1]])
patterns[13] = np.array([[1,1,1],[1,1,1],[-1,-1,-1]])

class RBM():
    
    def __init__(self, M, N):
        self.W = 2*np.random.rand(M, N) - 0.5
        self.threshold_h = 2*np.random.rand(M, 1) - 0.5
        self.threshold_v = 2*np.random.rand(N, 1) - 0.5
        self.N = N
        self.M = M
    
    def output(self, x, k, beta):
        v_0 = x
        b_h0 = self.W @ v_0 - self.threshold_h
        prob = p(b_h0, beta).flatten()
        h = np.array([[np.random.choice([-1, 1], p=[prob[i], 1-prob[i]]) for i in range(self.M)]]).T

        for t in range(k):
            b_v = h.T @ self.W - self.threshold_v
            prob = p(b_v, beta).flatten()
            v = np.array([[np.random.choice([-1, 1], p=[prob[i], 1-prob[i]]) for i in range(self.N)]]).T

            b_h = self.W @ v - self.threshold_h
            prob = p(b_h, beta).flatten() 
            h = np.array([[np.random.choice([-1, 1], p=[prob[i], 1-prob[i]]) for i in range(self.M)]]).T
        
        return v, b_h0, b_h

N = 9
Ms = [2, 4, 8, 16]
k = 100
LR = 0.1
beta = 1
nu_max = 3000
p_0 = 10
DKL_training = [[], [], [], []]
steps = [[], [], [], []]

p = lambda b, beta: 1 / (1 + np.exp(-2*beta*b))

def p_B(rbm, patterns, beta, nb_iterations=5000):
    distribution = np.zeros(14)
    for i in range(nb_iterations):
        x = np.random.choice([-1, 1], p=[0.5,0.5], size=(rbm.N, 1))
        output, b_h0, b_h = rbm.output(x, 10, beta)
        for idx, pattern in enumerate(patterns):   
            if (output.flatten() == pattern.flatten()).all():
                distribution[idx] += 1
                break
    distribution /= nb_iterations
    return distribution

def DKL(p_B_estimations):
    return sum([1/14 * np.log(1/14 / p_B_estimation) if p_B_estimation else 1/14 * np.log(1/14 / 0.0000001) for p_B_estimation in p_B_estimations])

for i, M in enumerate(Ms):   
    rbm = RBM(M, N)
    
    for nu in tqdm(range(nu_max)):    
        delta_W = np.zeros((M, N))
        delta_th_v = np.zeros((N, 1))
        delta_th_h = np.zeros((M, 1))
    
        rand_idx = np.random.choice(14, p_0, replace=False)
        patterns_sample = patterns[rand_idx]
            
        for mu in range(p_0):  
            x = patterns_sample[mu].reshape((1, N)).T
            v, b_h0, b_h = rbm.output(x, k, beta)
                        
            delta_W += LR * (np.tanh(b_h0) @ x.T - np.tanh(b_h) @ v.T)
            delta_th_v += -LR * (x - v)
            delta_th_h += -LR * (np.tanh(b_h0) - np.tanh(b_h))
            
        rbm.W += delta_W
        rbm.threshold_v += delta_th_v
        rbm.threshold_h += delta_th_h
        
        if nu%50 == 0 or nu == nu_max-1:
            p_B_estimations = p_B(rbm, patterns, beta)
            DKL_training[i].append(DKL(p_B_estimations)) 
            steps[i].append(nu) 
    
            print('For nu =', nu, 'M =', M, 'DKL =', DKL_training[i], p_B_estimations)
    
    np.save(f"W_{M}", rbm.W)
    np.save(f"threshold_v_{M}", rbm.threshold_v)
    np.save(f"threshold_h_{M}", rbm.threshold_h)
    
np.save("DKL_training", np.array(DKL_training))
np.save("steps", np.array(steps))


plt.plot(steps[0], DKL_training[0], label="M=2")
plt.plot(steps[1], DKL_training[1], label="M=4")
plt.plot(steps[2], DKL_training[2], label="M=8")
plt.plot(steps[3], DKL_training[3], label="M=16")
plt.title("DKL variation")
plt.ylabel("DKL")
plt.xlabel("Training steps")
plt.legend(loc="upper right")


# Let's keep the RBM with 16 hidden neurons as it is the best one
output, b_h0, b_h = rbm.output(np.array([[1,-1,-1],[0,0,0],[0,0,0]]), 10, beta)
print('Completed pattern', output)



