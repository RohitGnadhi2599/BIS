import numpy as np

population_size = 10
mutation_rate = 0.1
crossover_rate = 0.7
num_generations = 10
solution_length = 8

def fitness_function(individual):
    return sum(individual)

def create_initial_population():
    return np.random.randint(2, size=(population_size, solution_length))

def selection(population, fitness_scores):
    selected_indices = np.random.choice(range(population_size), size=population_size, p=fitness_scores/fitness_scores.sum())
    return population[selected_indices]

def crossover(parent1, parent2):
    if np.random.rand() < crossover_rate:
        crossover_point = np.random.randint(1, solution_length - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2
    return parent1, parent2

def mutation(individual):
    for i in range(solution_length):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

def genetic_algorithm():
    population = create_initial_population()
    best_solution = None
    best_fitness = -1

    for generation in range(num_generations):
        fitness_scores = np.array([fitness_function(individual) for individual in population])
        max_fitness_index = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_index] > best_fitness:
            best_fitness = fitness_scores[max_fitness_index]
            best_solution = population[max_fitness_index].copy()
        
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Best Solution = {best_solution}")
        population = selection(population, fitness_scores)
        next_population = []
        for i in range(0, population_size, 2):
            parent1, parent2 = population[i], population[i + 1 if i + 1 < population_size else 0]
            child1, child2 = crossover(parent1, parent2)
            next_population.append(mutation(child1))
            next_population.append(mutation(child2))
        population = np.array(next_population[:population_size])

    return best_solution, best_fitness

best_solution, best_fitness = genetic_algorithm()
print("\nBest Solution after 10 Generations:")
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
