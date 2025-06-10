import numpy as np
from scipy.optimize import minimize

class EnhancedHybridPSO_NelderMead:
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
        self.nelder_mead_budget_ratio = 0.2 #Fraction of budget for Nelder-Mead

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        if self.dim > 0:
            self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
            self.best_fitness_overall = objective_function(self.best_solution_overall.reshape(1,-1))[0]
            self.eval_count += 1
        else:
            self.best_solution_overall = np.array([])
            self.best_fitness_overall = 0


        swarm = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        personal_bests = swarm.copy()
        personal_best_fitness = np.array([objective_function(x.reshape(1,-1))[0] for x in swarm])
        self.eval_count += self.swarm_size

        nelder_mead_budget = int(self.budget * self.nelder_mead_budget_ratio)

        while self.eval_count < self.budget:
            for i in range(self.swarm_size):
                if personal_best_fitness[i] < self.best_fitness_overall:
                    self.best_fitness_overall = personal_best_fitness[i]
                    self.best_solution_overall = personal_bests[i].copy()

            # Adaptive inertia weight
            self.inertia_weight = 0.4 + 0.3 * (1 - (self.eval_count / self.budget))

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

            # Improved Nelder-Mead integration: only if significant improvement in PSO
            if self.eval_count < self.budget - nelder_mead_budget and self.best_fitness_overall < 0.8* np.mean(personal_best_fitness):
                res = minimize(objective_function, self.best_solution_overall, method='nelder-mead', options={'maxfev': nelder_mead_budget})
                if res.fun < self.best_fitness_overall and self.eval_count + res.nfev <= self.budget:
                    self.best_fitness_overall = res.fun
                    self.best_solution_overall = res.x
                    self.eval_count += res.nfev

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info