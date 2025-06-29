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
        self.swarm_size = int(np.ceil(self.dim * 5))  # Dynamic swarm size
        self.max_swarm_size = int(np.ceil(self.dim * 10)) # Maximum swarm size
        self.min_swarm_size = int(np.ceil(self.dim * 2)) # Minimum swarm size


    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
        self.best_fitness_overall = objective_function(self.best_solution_overall.reshape(1,-1))[0]
        self.eval_count += 1

        num_restarts = 5 #Number of restarts

        for restart in range(num_restarts):
            swarm = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.swarm_size, self.dim))
            velocities = np.zeros((self.swarm_size, self.dim))
            personal_bests = swarm.copy()
            personal_best_fitness = np.array([objective_function(x.reshape(1,-1))[0] for x in swarm])
            self.eval_count += self.swarm_size

            inertia_weight = 0.7 # Initial inertia weight
            while self.eval_count < self.budget:
                for i in range(self.swarm_size):
                    if personal_best_fitness[i] < self.best_fitness_overall:
                        self.best_fitness_overall = personal_best_fitness[i]
                        self.best_solution_overall = personal_bests[i].copy()

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities = inertia_weight * velocities + 1.4 * r1 * (personal_bests - swarm) + 1.4 * r2 * (self.best_solution_overall - swarm)
                swarm = swarm + velocities
                swarm = np.clip(swarm, self.lower_bounds, self.upper_bounds)

                fitness_values = objective_function(swarm)
                self.eval_count += self.swarm_size

                for i in range(self.swarm_size):
                    if fitness_values[i] < personal_best_fitness[i]:
                        personal_best_fitness[i] = fitness_values[i]
                        personal_bests[i] = swarm[i].copy()

                #Adaptive Swarm Size
                if np.mean(personal_best_fitness) < self.best_fitness_overall * 0.9 :
                    self.swarm_size = min(self.swarm_size * 2, self.max_swarm_size)
                elif np.mean(personal_best_fitness) > self.best_fitness_overall * 1.1 :
                    self.swarm_size = max(self.swarm_size // 2, self.min_swarm_size)


                #Adaptive Inertia Weight
                inertia_weight = 0.4 + 0.3 * np.exp(-self.eval_count / self.budget)

                # Local Search with Nelder-Mead (triggered more strategically)
                if self.eval_count < 0.8 * self.budget and self.eval_count % (self.swarm_size * 2) == 0: #Trigger Local search every other iteration
                    res = minimize(objective_function, self.best_solution_overall, method='nelder-mead', options={'maxfev': int(0.02 * self.budget)}) #Reduced maxfev for better efficiency
                    if res.fun < self.best_fitness_overall and self.eval_count + res.nfev <= self.budget:
                        self.best_fitness_overall = res.fun
                        self.best_solution_overall = res.x
                        self.eval_count += res.nfev

                if self.eval_count >= self.budget:
                    break

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info