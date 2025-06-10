import numpy as np
from scipy.optimize import minimize

class AdaptiveHybridPSONelderMead:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.swarm_size = int(np.sqrt(self.budget))  # Initial swarm size
        self.inertia_weight = 0.7
        self.cognitive_weight = 1.4
        self.social_weight = 1.4
        self.adaptive_factor = 0.95 #Factor to reduce swarm size

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
        self.best_fitness_overall = objective_function(self.best_solution_overall.reshape(1,-1))[0]
        self.eval_count += 1

        swarm = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.swarm_size, self.dim))
        velocities = np.zeros_like(swarm)

        personal_bests = swarm.copy()
        personal_best_fitnesses = objective_function(personal_bests)
        self.eval_count += self.swarm_size

        for i in range(min(self.budget, 1000)):  #Iteration limit for stability
            for j in range(self.swarm_size):
                if personal_best_fitnesses[j] < self.best_fitness_overall:
                    self.best_solution_overall = personal_bests[j].copy()
                    self.best_fitness_overall = personal_best_fitnesses[j]

            r1 = np.random.random(self.dim)
            r2 = np.random.random(self.dim)

            velocities = self.inertia_weight * velocities + self.cognitive_weight * r1 * (personal_bests - swarm) + self.social_weight * r2 * (self.best_solution_overall - swarm)
            swarm = swarm + velocities

            swarm = np.clip(swarm, self.lower_bounds, self.upper_bounds)

            fitness_values = objective_function(swarm)
            self.eval_count += self.swarm_size

            for k in range(self.swarm_size):
                if fitness_values[k] < personal_best_fitnesses[k]:
                    personal_bests[k] = swarm[k].copy()
                    personal_best_fitnesses[k] = fitness_values[k]

            # Adaptive Inertia Weight
            self.inertia_weight *= self.adaptive_factor


            # Adaptive Swarm Size Reduction
            if i % 10 == 0 and self.swarm_size > 10 and self.eval_count < self.budget * 0.8 : #Reduce swarm size over time
              self.swarm_size = int(self.swarm_size * self.adaptive_factor)
              swarm = swarm[:self.swarm_size,:]
              personal_bests = personal_bests[:self.swarm_size,:]
              personal_best_fitnesses = personal_best_fitnesses[:self.swarm_size]
              velocities = velocities[:self.swarm_size,:]


            # Nelder-Mead local search around the global best
            if self.eval_count < self.budget:
                result = minimize(objective_function, self.best_solution_overall, method='Nelder-Mead', options={'maxfev': min(100, self.budget - self.eval_count)})
                if result.fun < self.best_fitness_overall:
                    self.best_solution_overall = result.x
                    self.best_fitness_overall = result.fun
                self.eval_count += result.nfev

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info