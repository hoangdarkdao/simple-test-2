import numpy as np
import random

class EnhancedArchiveGuidedDE: #aocc 0.15
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float],
                 population_size_factor: float = 8.82865217019506, archive_size: int = 165.22481375900153, initial_F_scale: float = 0.3544373580018585):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')

        self.population_size = int(population_size_factor * self.dim)  # common heuristic
        self.archive_size = archive_size
        self.archive = []
        self.population = None
        self.F_scale = initial_F_scale  # initial scaling factor

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8,
                 F_scale_variation: float = 0.3, archive_update_threshold: float = 0.8) -> tuple:
        self.eval_count = 0
        self.population = np.random.uniform(self.lower_bounds, self.upper_bounds, size=(self.population_size, self.dim))
        fitness = objective_function(self.population)
        self.eval_count += self.population_size

        self.best_solution_overall = self.population[np.argmin(fitness)]
        self.best_fitness_overall = np.min(fitness)

        while self.eval_count < self.budget:
            offspring = self.generate_offspring(self.population, fitness, F_scale_variation)
            offspring_fitness = objective_function(offspring)
            self.eval_count += len(offspring)

            # Update archive
            self.update_archive(offspring, offspring_fitness, archive_update_threshold)

            # Select best solutions for next generation
            combined_population = np.concatenate((self.population, offspring))
            combined_fitness = np.concatenate((fitness, offspring_fitness))
            indices = np.argsort(combined_fitness)
            self.population = combined_population[indices[:self.population_size]]
            fitness = combined_fitness[indices[:self.population_size]]

            # Update best solution
            self.best_solution_overall = self.population[np.argmin(fitness)]
            self.best_fitness_overall = np.min(fitness)

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall,
            'archive_size': len(self.archive)
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info

    def generate_offspring(self, population, fitness, F_scale_variation):
        offspring = np.zeros((self.population_size, self.dim))
        # Adaptive scaling factor
        self.F_scale = 0.5 + F_scale_variation * np.random.rand()  # scale factor with slight variation

        for i in range(self.population_size):
            # Select pbest from archive (if available)
            if self.archive:
                pbest_index = np.random.choice(len(self.archive))
                pbest = self.archive[pbest_index][0]
            else:
                pbest = population[np.argmin(fitness)]

            a, b, c = random.sample(range(self.population_size), 3)
            while a == i or b == i or c == i:
                a, b, c = random.sample(range(self.population_size), 3)

            offspring[i] = population[i] + self.F_scale * (pbest - population[i] + population[a] - population[b])
            offspring[i] = np.clip(offspring[i], self.lower_bounds, self.upper_bounds)  # Boundary handling

        return offspring

    def update_archive(self, offspring, offspring_fitness, archive_update_threshold):
        for i in range(len(offspring)):
            if len(self.archive) < self.archive_size:
                self.archive.append((offspring[i], offspring_fitness[i]))
            else:
                # Prioritize diversity in archive
                worst_index = np.argmax([f for _, f in self.archive])
                if offspring_fitness[i] < self.archive[worst_index][1] or len(self.archive) < self.archive_size * archive_update_threshold:
                    self.archive[worst_index] = (offspring[i], offspring_fitness[i])