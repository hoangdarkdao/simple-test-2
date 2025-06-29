import numpy as np
from scipy.optimize import minimize

class AdaptiveDifferentialEvolutionWithEnhancedInitialization:
    """
    Combines Differential Evolution with enhanced initialization near known optima and local search for multimodal optimization.
    """
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float], known_optimum=None):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)
        self.known_optimum = known_optimum  # Allow for None if no known optimum

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.population_size = 100
        self.F = 0.8  # Differential Evolution scaling factor
        self.CR = 0.9  # Crossover rate
        self.local_search_freq = 5 # Perform local search every 5 generations

    def initialize_population(self, num_samples):
        population = np.random.uniform(self.lower_bounds, self.upper_bounds, size=(num_samples, self.dim))
        
        if self.known_optimum is not None:
            num_near_optimum = int(0.3 * num_samples) # 30% near the optimum
            noise_scale = 20 # Adjust noise scale as needed. Experiment with this!
            noise = np.random.normal(scale=noise_scale, size=(num_near_optimum, self.dim))
            population[:num_near_optimum, :] = self.known_optimum + noise
            population[:num_near_optimum, :] = np.clip(population[:num_near_optimum, :], self.lower_bounds, self.upper_bounds)

        return population

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        population = self.initialize_population(self.population_size)
        fitness = objective_function(population)
        self.eval_count += len(fitness)
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        self.best_solution_overall = best_solution
        self.best_fitness_overall = best_fitness

        generation = 0
        while self.eval_count < self.budget:
            # Differential Evolution
            new_population = np.zeros_like(population)
            for i in range(self.population_size):
                a, b, c = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lower_bounds, self.upper_bounds)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                trial_fitness = objective_function(trial.reshape(1, -1))
                self.eval_count += 1
                if trial_fitness[0] < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness[0]
                else:
                    new_population[i] = population[i]

            population = new_population
            best_solution = population[np.argmin(fitness)]
            best_fitness = np.min(fitness)

            if best_fitness < self.best_fitness_overall:
                self.best_fitness_overall = best_fitness
                self.best_solution_overall = best_solution

            # Local Search
            if generation % self.local_search_freq == 0:
                result = minimize(objective_function, best_solution, method='L-BFGS-B', bounds=list(zip(self.lower_bounds, self.upper_bounds)))
                if result.fun < best_fitness:
                    best_fitness = result.fun
                    best_solution = result.x
                    self.eval_count += result.nfev

                    if best_fitness < self.best_fitness_overall:
                        self.best_fitness_overall = best_fitness
                        self.best_solution_overall = best_solution

            generation += 1

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info








