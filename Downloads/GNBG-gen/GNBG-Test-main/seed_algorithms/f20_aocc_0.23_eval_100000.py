import numpy as np

class AdaptiveGaussianSamplingEA:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.population_size = 100  # Adjust as needed
        self.population = None
        self.sigma = 0.2 * (self.upper_bounds - self.lower_bounds)  # Initial Standard Deviation for Gaussian Sampling

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        self.population = np.random.uniform(self.lower_bounds, self.upper_bounds, size=(self.population_size, self.dim))
        fitness_values = objective_function(self.population)
        self.eval_count += self.population_size

        self.best_solution_overall = self.population[np.argmin(fitness_values)]
        self.best_fitness_overall = np.min(fitness_values)

        while self.eval_count < self.budget:
            # Adaptive Gaussian Sampling
            parents = self.tournament_selection(fitness_values, k=5)  # Tournament Selection
            offspring = self.gaussian_mutation(parents, self.sigma)

            # Bounds handling
            offspring = np.clip(offspring, self.lower_bounds, self.upper_bounds)

            offspring_fitness = objective_function(offspring)
            self.eval_count += len(offspring)

            # Update population and best solution
            self.population = np.concatenate((self.population, offspring))
            fitness_values = np.concatenate((fitness_values, offspring_fitness))

            best_index = np.argmin(fitness_values)
            if fitness_values[best_index] < self.best_fitness_overall:
                self.best_solution_overall = self.population[best_index]
                self.best_fitness_overall = fitness_values[best_index]

            # Adaptive Sigma
            self.sigma *= 0.99  # Gradually reduce sigma for finer search later.

            # Elitism
            sorted_pop = self.population[np.argsort(fitness_values)]
            self.population = sorted_pop[:self.population_size]
            fitness_values = fitness_values[np.argsort(fitness_values)][:self.population_size]

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info

    def tournament_selection(self, fitnesses, k):
        num_parents = len(fitnesses) // 2  # Select half the population as parents
        parents = np.zeros((num_parents, self.dim))
        for i in range(num_parents):
            tournament = np.random.choice(len(fitnesses), size=k, replace=False)
            winner_index = tournament[np.argmin(fitnesses[tournament])]
            parents[i] = self.population[winner_index]
        return parents

    def gaussian_mutation(self, parents, sigma):
        offspring = parents + np.random.normal(0, sigma, parents.shape)
        return offspring
