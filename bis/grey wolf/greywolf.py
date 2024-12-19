# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 22:11:13 2024

@author: R.N.GANDHI
"""
import numpy as np
import random

def fitness_function(matrix):
    return np.sum(np.square(matrix))

def update_position(wolf_position, alpha_position, beta_position, delta_position, a, b):
    r1 = np.random.rand()
    r2 = np.random.rand()
    A1 = 2 * a * r1 - a
    C1 = 2 * r2
    D_alpha = np.abs(C1 * alpha_position - wolf_position)
    X1 = alpha_position - A1 * D_alpha

    r1 = np.random.rand()
    r2 = np.random.rand()
    A2 = 2 * a * r1 - a
    C2 = 2 * r2
    D_beta = np.abs(C2 * beta_position - wolf_position)
    X2 = beta_position - A2 * D_beta

    r1 = np.random.rand()
    r2 = np.random.rand()
    A3 = 2 * a * r1 - a
    C3 = 2 * r2
    D_delta = np.abs(C3 * delta_position - wolf_position)
    X3 = delta_position - A3 * D_delta

    return (X1 + X2 + X3) / 3

def gwo_optimization(matrix, max_iter=10, num_wolves=10):
    alpha_position = np.random.rand(*matrix.shape)
    beta_position = np.random.rand(*matrix.shape)
    delta_position = np.random.rand(*matrix.shape)

    alpha_score = fitness_function(alpha_position)
    beta_score = fitness_function(beta_position)
    delta_score = fitness_function(delta_position)

    wolves = np.random.rand(num_wolves, *matrix.shape)
    a = 2
    for t in range(max_iter):
        for i in range(num_wolves):
            wolf_position = wolves[i]
            fitness = fitness_function(wolf_position)

            if fitness < alpha_score:
                alpha_score = fitness
                alpha_position = wolf_position
            elif fitness < beta_score:
                beta_score = fitness
                beta_position = wolf_position
            elif fitness < delta_score:
                delta_score = fitness
                delta_position = wolf_position

        a = 2 - t * (2 / max_iter)
        for i in range(num_wolves):
            wolves[i] = update_position(wolves[i], alpha_position, beta_position, delta_position, a, i)

        print(f"Iteration {t+1}: Alpha Position:\n{alpha_position}\nAlpha Score: {alpha_score}")
        print(f"Iteration {t+1}: Beta Position:\n{beta_position}\nBeta Score: {beta_score}")
        print(f"Iteration {t+1}: Delta Position:\n{delta_position}\nDelta Score: {delta_score}\n")

    return alpha_position

matrix = np.random.rand(5, 5)
optimized_matrix = gwo_optimization(matrix)
print("Optimized Matrix:\n", optimized_matrix)
