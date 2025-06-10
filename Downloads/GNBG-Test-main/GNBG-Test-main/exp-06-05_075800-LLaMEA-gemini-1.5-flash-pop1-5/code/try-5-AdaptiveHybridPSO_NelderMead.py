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
        self.swarm_size = int(np.ceil(self.dim * 5))  # Dynamic swarm size
        self.inertia_weight = 0.7
        self.cognitive_factor = 1.4
        self.social_factor = 1.4
        self.adaptive_factor = 0.98 #Factor to decrease inertia weight

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        if self.dim > 0:
            self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
            self.best_fitness_overall = objective_function(self.best_solution_overall.reshape(1,-1))[0]
            self.eval_count += 1
        else:
            self.best_solution_overall = np.array([])
            self.best_fitness_overall = 0.0 #Handle 0 dimension case


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

            # Adaptive Inertia Weight
            self.inertia_weight *= self.adaptive_factor

            # Adaptive Swarm Size (increase if stuck)
            if self.eval_count > self.budget * 0.7 and self.inertia_weight < 0.1 : #Avoid unnecessary increase at end
                self.swarm_size = int(self.swarm_size * 1.5)
                swarm = np.concatenate((swarm, np.random.uniform(self.lower_bounds, self.upper_bounds, (self.swarm_size - swarm.shape[0], self.dim))))
                velocities = np.concatenate((velocities, np.zeros((self.swarm_size - velocities.shape[0], self.dim))))
                personal_bests = np.concatenate((personal_bests, swarm[-self.swarm_size + swarm.shape[0]:]))
                personal_best_fitness = np.concatenate((personal_best_fitness, [objective_function(x.reshape(1, -1))[0] for x in swarm[-self.swarm_size + swarm.shape[0]:]]))
                self.eval_count += self.swarm_size - swarm.shape[0]


            # Local Search with Nelder-Mead
            res = minimize(objective_function, self.best_solution_overall, method='nelder-mead', options={'maxfev': int(100 + (self.budget - self.eval_count) * 0.2)}) #dynamic maxfev
            if res.fun < self.best_fitness_overall and self.eval_count + res.nfev <= self.budget:
                self.best_fitness_overall = res.fun
                self.best_solution_overall = res.x
                self.eval_count += res.nfev
            if self.eval_count >= self.budget: break

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info
