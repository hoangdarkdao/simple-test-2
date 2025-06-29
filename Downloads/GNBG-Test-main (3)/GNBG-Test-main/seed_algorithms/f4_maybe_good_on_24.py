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
        self.sigma = 0.2 * (self.upper_bounds - self.lower_bounds)  # Initial Gaussian width
        self.sigma_decay = 0.99 #Decay rate of sigma


    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
        self.best_fitness_overall = objective_function(self.best_solution_overall.reshape(1, -1))[0]
        self.eval_count += 1

        population = self._initialize_population()
        fitness_values = objective_function(population)
        self.eval_count += self.population_size

        while self.eval_count < self.budget:
            # Selection (tournament selection)
            parents = self._tournament_selection(population, fitness_values)

            # Recombination (Gaussian perturbation)
            offspring = self._gaussian_recombination(parents)

            # Mutation (adjust sigma adaptively)
            offspring = self._adaptive_mutation(offspring)

            #Evaluation
            offspring_fitness = objective_function(offspring)
            self.eval_count += len(offspring)
            
            #Selection for next generation
            population, fitness_values = self._select_next_generation(population, fitness_values, offspring, offspring_fitness)

            # Update best solution
            self._update_best(offspring, offspring_fitness)
            self.sigma *= self.sigma_decay

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info


    def _initialize_population(self):
        #Adaptive Gaussian Sampling
        center = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
        population = np.random.normal(center, self.sigma, size=(self.population_size, self.dim))
        population = np.clip(population, self.lower_bounds, self.upper_bounds)
        return population

    def _tournament_selection(self, population, fitness_values):
        tournament_size = 5
        num_parents = self.population_size //2 #Binary Recombination

        selected_parents = []
        for _ in range(num_parents):
            tournament = np.random.choice(len(population), tournament_size, replace=False)
            winner_index = tournament[np.argmin(fitness_values[tournament])]
            selected_parents.append(population[winner_index])
        return np.array(selected_parents)


    def _gaussian_recombination(self, parents):
        offspring = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i+1]
            child1 = (parent1 + parent2) / 2 + np.random.normal(0, self.sigma / 2, self.dim)
            child2 = (parent1 + parent2) / 2 + np.random.normal(0, self.sigma / 2, self.dim)
            offspring.extend([child1, child2])
        return np.clip(np.array(offspring), self.lower_bounds, self.upper_bounds)


    def _adaptive_mutation(self, offspring):
        #Simple Mutation, sigma is already updated
        offspring += np.random.normal(0, self.sigma, size=offspring.shape)
        return np.clip(offspring, self.lower_bounds, self.upper_bounds)

    def _select_next_generation(self, population, fitness_values, offspring, offspring_fitness):
        combined_pop = np.vstack((population, offspring))
        combined_fit = np.concatenate((fitness_values, offspring_fitness))
        
        sorted_indices = np.argsort(combined_fit)
        
        next_gen = combined_pop[sorted_indices[:self.population_size]]
        next_fit = combined_fit[sorted_indices[:self.population_size]]
        return next_gen, next_fit
    

    def _update_best(self, offspring, offspring_fitness):
        for i, fitness in enumerate(offspring_fitness):
            if fitness < self.best_fitness_overall:
                self.best_fitness_overall = fitness
                self.best_solution_overall = offspring[i]