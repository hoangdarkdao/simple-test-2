import numpy as np
import random
# f18 aocc 0.8
# f20 aocc 0.5
# not so good again, get stuck in local optima
class AdaptivePopulationDE: 
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')

        self.population_size = 10 * self.dim
        self.min_population_size = 5 * self.dim
        self.max_population_size = 20 * self.dim
        self.population_adaptation_rate = 0.1

        self.F = 0.5  # Mutation factor
        self.Cr = 0.7 # Crossover rate

        self.stagnation_counter = 0
        self.stagnation_threshold = 5000

        self.archive = []
        self.archive_size = 100

        self.population = None
        self.fitness = None

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        self.stagnation_counter = 0

        self.population = np.random.uniform(self.lower_bounds, self.upper_bounds, size=(self.population_size, self.dim))
        self.fitness = objective_function(self.population)
        self.eval_count += self.population_size

        best_index = np.argmin(self.fitness)
        self.best_solution_overall = self.population[best_index]
        self.best_fitness_overall = self.fitness[best_index]

        while self.eval_count < self.budget:
            offspring = self.generate_offspring(objective_function)
            offspring_fitness = objective_function(offspring)
            self.eval_count += len(offspring)

            self.update_archive(offspring, offspring_fitness)

            for i in range(self.population_size):
                if offspring_fitness[i] < self.fitness[i]:
                    self.population[i] = offspring[i]
                    self.fitness[i] = offspring_fitness[i]

            best_index = np.argmin(self.fitness)
            if self.fitness[best_index] < self.best_fitness_overall:
                self.best_solution_overall = self.population[best_index]
                self.best_fitness_overall = self.fitness[best_index]
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += len(offspring)

            self.adjust_population_size(objective_function)

            if self.stagnation_counter > self.stagnation_threshold:
                self.restart_population(objective_function)
                self.stagnation_counter = 0

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall,
            'population_size': self.population_size
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info

    def generate_offspring(self, objective_function):
        offspring = np.zeros((self.population_size, self.dim))

        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            if len(indices) < 2:
                continue  # Skip if not enough individuals

            a, b = random.sample(indices, 2)

            if self.archive and random.random() < 0.5:
                pbest = self.archive[random.randint(0, len(self.archive) - 1)][0]
            else:
                pbest = self.population[np.argmin(self.fitness)]

            mutant = self.population[i] + self.F * (pbest - self.population[i] + self.population[a] - self.population[b])

            for j in range(self.dim):
                if random.random() > self.Cr:
                    mutant[j] = self.population[i][j]

            offspring[i] = np.clip(mutant, self.lower_bounds, self.upper_bounds)

        return offspring

    def update_archive(self, offspring, offspring_fitness):
        for i in range(len(offspring)):
            if len(self.archive) < self.archive_size:
                self.archive.append((offspring[i], offspring_fitness[i]))
            else:
                worst_index = np.argmax([f for _, f in self.archive])
                if offspring_fitness[i] < self.archive[worst_index][1] or len(self.archive) < self.archive_size * 0.8:
                    self.archive[worst_index] = (offspring[i], offspring_fitness[i])

    def adjust_population_size(self, objective_function):
        if random.random() < self.population_adaptation_rate:
            if self.stagnation_counter > self.stagnation_threshold / 2:
                new_size = min(int(self.population_size * 1.1), self.max_population_size)
            else:
                new_size = max(int(self.population_size * 0.9), self.min_population_size)

            new_size = int(new_size)
            if new_size > self.population_size:
                additional = np.random.uniform(self.lower_bounds, self.upper_bounds, size=(new_size - self.population_size, self.dim))
                additional_fitness = objective_function(additional)
                self.population = np.vstack((self.population, additional))
                self.fitness = np.concatenate((self.fitness, additional_fitness))
                self.eval_count += len(additional)
            elif new_size < self.population_size:
                best_indices = np.argsort(self.fitness)[:new_size]
                self.population = self.population[best_indices]
                self.fitness = self.fitness[best_indices]

            self.population_size = new_size

    def restart_population(self, objective_function):
        self.population = np.random.uniform(self.lower_bounds, self.upper_bounds, size=(self.population_size, self.dim))
        self.fitness = objective_function(self.population)
        self.eval_count += self.population_size
        best_index = np.argmin(self.fitness)
        if self.fitness[best_index] < self.best_fitness_overall:
            self.best_solution_overall = self.population[best_index]
            self.best_fitness_overall = self.fitness[best_index]