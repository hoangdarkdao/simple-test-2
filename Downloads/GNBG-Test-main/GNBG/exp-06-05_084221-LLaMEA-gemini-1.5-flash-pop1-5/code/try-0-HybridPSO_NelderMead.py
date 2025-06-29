import numpy as np
from scipy.optimize import minimize

class HybridPSO_NelderMead:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None 
        self.best_fitness_overall = float('inf')
        self.swarm_size = int(np.sqrt(self.budget)) # adaptive swarm size
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.4
        self.social_coeff = 1.4

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        if self.dim > 0:
            self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
            self.best_fitness_overall = objective_function(self.best_solution_overall.reshape(1,-1))[0]
            self.eval_count +=1
        else:
            self.best_solution_overall = np.array([])
            self.best_fitness_overall = 0

        swarm = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.swarm_size, self.dim))
        velocities = np.zeros_like(swarm)
        personal_bests = swarm.copy()
        personal_best_fitness = np.array([objective_function(x.reshape(1,-1))[0] for x in swarm])
        self.eval_count += self.swarm_size

        for i in range(int(self.budget/self.swarm_size)):
            # Update Velocities and Positions
            r1 = np.random.random((self.swarm_size, self.dim))
            r2 = np.random.random((self.swarm_size, self.dim))
            velocities = self.inertia_weight * velocities + self.cognitive_coeff * r1 * (personal_bests - swarm) + self.social_coeff * r2 * (self.best_solution_overall - swarm)
            swarm = swarm + velocities
            
            #Clamp to bounds
            swarm = np.clip(swarm, self.lower_bounds, self.upper_bounds)
            
            # Evaluate fitness
            fitness_values = objective_function(swarm)
            self.eval_count += self.swarm_size

            #Update personal bests
            better_indices = fitness_values < personal_best_fitness
            personal_bests[better_indices] = swarm[better_indices]
            personal_best_fitness[better_indices] = fitness_values[better_indices]


            # Update global best
            min_index = np.argmin(personal_best_fitness)
            if personal_best_fitness[min_index] < self.best_fitness_overall:
                self.best_solution_overall = personal_bests[min_index].copy()
                self.best_fitness_overall = personal_best_fitness[min_index]


        #Local Search with Nelder-Mead
        res = minimize(objective_function, self.best_solution_overall, method='Nelder-Mead', options={'maxiter': int(self.budget*0.1), 'disp': False})
        self.best_solution_overall = res.x
        self.best_fitness_overall = res.fun
        self.eval_count += res.nfev
        

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info
