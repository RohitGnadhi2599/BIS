# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 22:25:19 2024

@author: R.N.GANDHI
"""

import numpy as np
from scipy.special import gamma

def fitness_function(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    contrast = max_val - min_val
    color_variance = np.var(matrix)
    return contrast + color_variance

def levy_flight(Lambda):
    beta = 3 / 2
    sigma_u = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma_u, 1)
    v = np.random.normal(0, 1, 1)
    step = u / (abs(v) ** (1 / beta))
    return step

def cuckoo_search(matrix, num_nests=10, max_iter=10, pa=0.25):
    nests = np.random.rand(num_nests, *matrix.shape)
    best_nest = nests[0]
    best_fitness = fitness_function(best_nest)
    
    for t in range(max_iter):
        for i in range(num_nests):
            new_nest = nests[i] + levy_flight(1.5) * (nests[i] - best_nest)
            new_nest = np.clip(new_nest, 0, 1)
            new_fitness = fitness_function(new_nest)
            
            if new_fitness > best_fitness:
                best_nest = new_nest
                best_fitness = new_fitness
        
        for i in range(num_nests):
            if np.random.rand() < pa:
                nests[i] = np.random.rand(*matrix.shape)
        
        print(f"Iteration {t + 1}: Best Nest:\n{best_nest}")
        print(f"Iteration {t + 1}: Best Fitness: {best_fitness}\n")
    
    return best_nest

matrix = np.random.rand(5, 5)
optimized_matrix = cuckoo_search(matrix)
print("Optimized Matrix:\n", optimized_matrix)

