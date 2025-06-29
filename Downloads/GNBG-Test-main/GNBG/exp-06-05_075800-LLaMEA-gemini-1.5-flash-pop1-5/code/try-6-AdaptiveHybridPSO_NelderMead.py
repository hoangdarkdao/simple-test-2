import numpy as np
from scipy.optimize import minimize

class AdaptiveHybridPSO_NelderMead:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.swarm_size = int(np.ceil(self.dim * 5)) # Initial swarm size
        self.inertia_weight = 0.7
        self.cognitive_factor = 1.4
        self.social_factor = 1.4
        self.exploration_factor = 0.9 # Adjust exploration vs exploitation


    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        if self.dim > 0:
            self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
            self.best_fitness_overall = objective_function(self.best_solution_overall.reshape(1,-1))[0]
            self.eval_count += 1
        else:
            self.best_solution_overall = np.array([])
            self.best_fitness_overall = 0.0


        swarm = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        personal_bests = swarm.copy()
        personal_best_fitness = np.array([objective_function(x.reshape(1,-1))[0] for x in swarm])
        self.eval_count += self.swarm_size

        while self.eval_count < self.budget:
            for i in range(self.swarm_size):
                if personal_best_fitness[i] < self.best_fitness_overall:
                    self.best_fitness_overall = personal_best_fitness[i]
                    self.best_solution_overall = personal_bests[i].copy()

            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            velocities = self.inertia_weight * velocities + self.cognitive_factor * r1 * (personal_bests - swarm) + self.social_factor * r2 * (self.best_solution_overall - swarm)
            swarm = swarm + velocities
            swarm = np.clip(swarm, self.lower_bounds, self.upper_bounds)

            fitness_values = objective_function(swarm)
            self.eval_count += self.swarm_size

            for i in range(self.swarm_size):
                if fitness_values[i] < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness_values[i]
                    personal_bests[i] = swarm[i].copy()

            # Adaptive adjustments
            improvement_rate = (self.best_fitness_overall - np.mean(personal_best_fitness)) / self.best_fitness_overall if self.best_fitness_overall !=0 else 1.0
            if improvement_rate < 0.1: # Needs more exploration
                self.swarm_size = int(np.ceil(self.swarm_size * self.exploration_factor))
                self.inertia_weight = max(0.4, self.inertia_weight * 0.9) # Increase inertia
            else: # Needs more exploitation
                self.swarm_size = int(np.ceil(self.swarm_size / self.exploration_factor))
                self.inertia_weight = min(0.9, self.inertia_weight * 1.1) # Decrease inertia

            self.swarm_size = max(1, min(self.swarm_size, self.budget - self.eval_count)) #Avoid exceeding budget

            # Local Search with Nelder-Mead
            if self.eval_count < self.budget:
                res = minimize(objective_function, self.best_solution_overall, method='nelder-mead', options={'maxfev': min(100, self.budget - self.eval_count)})
                if res.fun < self.best_fitness_overall:
                    self.best_fitness_overall = res.fun
                    self.best_solution_overall = res.x
                    self.eval_count += res.nfev

            if self.eval_count >= self.budget: break

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info
