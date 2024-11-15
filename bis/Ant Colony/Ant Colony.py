# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 06:35:45 2024

@author: R.N.GANDHI
"""

import numpy as np

num_cities = 10
num_ants = 5
num_iterations = 10
alpha = 1.0
beta = 2.0
rho = 0.5
initial_pheromone = 1.0

cities = np.random.rand(num_cities, 2) * 100
distances = np.sqrt((np.square(cities[:, np.newaxis] - cities).sum(axis=2)))
pheromones = np.full((num_cities, num_cities), initial_pheromone)

def calculate_probabilities(ant_position, visited):
    heuristic = 1.0 / (distances[ant_position] + np.eye(num_cities)[ant_position])
    probabilities = (pheromones[ant_position] ** alpha) * (heuristic ** beta)
    probabilities[visited] = 0
    return probabilities / probabilities.sum()

def construct_solution():
    solution = [np.random.randint(num_cities)]
    for _ in range(num_cities - 1):
        probabilities = calculate_probabilities(solution[-1], solution)
        next_city = np.random.choice(range(num_cities), p=probabilities)
        solution.append(next_city)
    return solution

def calculate_solution_length(solution):
    return sum(distances[solution[i - 1], solution[i]] for i in range(num_cities))

def update_pheromones(all_solutions):
    global pheromones
    pheromones *= (1 - rho)
    for solution, length in all_solutions:
        for i in range(num_cities):
            pheromones[solution[i - 1], solution[i]] += 1.0 / length

best_solution = None
best_length = float('inf')

for iteration in range(num_iterations):
    all_solutions = []
    for _ in range(num_ants):
        solution = construct_solution()
        length = calculate_solution_length(solution)
        all_solutions.append((solution, length))
        if length < best_length:
            best_solution = solution
            best_length = length
    update_pheromones(all_solutions)
    print(f"Iteration {iteration + 1}: Best Length = {best_length}, Best Solution = {best_solution}")

print("\nBest Solution after Iterations:")
print("Best Solution:", best_solution)
print("Best Length:", best_length)
