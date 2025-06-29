import numpy as np
from scipy.optimize import minimize

class EnhancedHybridPSONelderMeadImprovedV3:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.population_size = 20 # Parameter for PSO
        self.inertia_weight = 0.7 # PSO parameter
        self.cognitive_coefficient = 1.4 # PSO parameter
        self.social_coefficient = 1.4 # PSO parameter

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
        self.best_fitness_overall = objective_function(self.best_solution_overall.reshape(1,-1))[0]
        self.eval_count +=1
        
        population = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_bests = population.copy()
        personal_best_fitness = np.apply_along_axis(lambda x: objective_function(x.reshape(1,-1))[0], 1, population)
        self.eval_count += self.population_size

        global_best = self.best_solution_overall.copy()
        
        while self.eval_count < self.budget:
            for i in range(self.population_size):
                if personal_best_fitness[i] < self.best_fitness_overall:
                    self.best_fitness_overall = personal_best_fitness[i]
                    self.best_solution_overall = personal_bests[i].copy()
                    global_best = self.best_solution_overall.copy() # Update global best immediately
                    
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            # Adaptive inertia weight
            self.inertia_weight = 0.4 + 0.3 * np.exp(-self.eval_count / self.budget)
            velocities = self.inertia_weight * velocities + self.cognitive_coefficient * r1 * (personal_bests - population) + self.social_coefficient * r2 * (global_best - population)
            population = population + velocities
            
            population = np.clip(population, self.lower_bounds, self.upper_bounds)

            #Cauchy Mutation for diversification - Improved scaling
            cauchy_mutation = np.random.standard_cauchy(size=(self.population_size, self.dim)) * (self.upper_bounds - self.lower_bounds) * 0.02 # Refined scaling
            population = population + cauchy_mutation
            population = np.clip(population, self.lower_bounds, self.upper_bounds)


            fitness_values = objective_function(population)
            self.eval_count += self.population_size
            
            for i in range(self.population_size):
                if fitness_values[i] < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness_values[i]
                    personal_bests[i] = population[i].copy()


            #Local search with Nelder-Mead - Increased iterations
            res = minimize(objective_function, self.best_solution_overall, method='Nelder-Mead', options={'maxiter': 20, 'maxfev': 20*self.dim}) #Increased maxiter and maxfev
            if res.fun < self.best_fitness_overall and self.eval_count + res.nfev <= self.budget:
                self.best_fitness_overall = res.fun
                self.best_solution_overall = res.x
                self.eval_count += res.nfev

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info