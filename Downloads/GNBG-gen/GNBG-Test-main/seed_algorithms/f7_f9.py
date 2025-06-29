import numpy as np
import random
class AdaptiveMultimodalOptimizerImproved:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.tabu_list = []  # Tabu list to avoid revisiting recent solutions
        self.tabu_length = 10 # Length of the tabu list

        self.perturbation_strength = 0.5 # Initial perturbation strength, adaptive
        self.local_search_iterations = 10 # Number of iterations for local search
        self.temperature = 1.0 # Initial temperature for simulated annealing

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        if self.dim > 0:
            self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
        else:
            self.best_solution_overall = np.array([])

        self.best_fitness_overall = objective_function(self.best_solution_overall.reshape(1, -1))[0]
        self.eval_count += 1

        while self.eval_count < self.budget:
            current_solution = self.best_solution_overall.copy()
                                
            # Local Search
            for _ in range(self.local_search_iterations):
                neighbor = self._generate_neighbor(current_solution)
                neighbor_fitness = objective_function(neighbor.reshape(1, -1))[0]
                self.eval_count += 1

                if self._accept(neighbor_fitness, self._fitness(current_solution, objective_function), self.temperature):
                    current_solution = neighbor

                if neighbor_fitness < self.best_fitness_overall:
                    self.best_fitness_overall = neighbor_fitness
                    self.best_solution_overall = neighbor
                    self.tabu_list = [] # Reset tabu list upon finding a new global best

            # Check for stagnation and apply perturbation
            if self._is_stagnant(current_solution):
                current_solution = self._perturb(current_solution)
            self.temperature *= 0.95 # Cool down the temperature

            # Add solution to tabu list
            self._update_tabu_list(current_solution)

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall,
            'perturbation_strength': self.perturbation_strength,
            'final_temperature': self.temperature
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info


    def _generate_neighbor(self, solution):
        neighbor = solution.copy()
        index = random.randint(0, self.dim - 1)
        neighbor[index] += np.random.normal(0, 0.1 * (self.upper_bounds[index] - self.lower_bounds[index]))  # Small Gaussian perturbation
        neighbor = np.clip(neighbor, self.lower_bounds, self.upper_bounds)
        return neighbor

    def _perturb(self, solution):
        perturbation = np.random.uniform(-self.perturbation_strength, self.perturbation_strength, self.dim) * (self.upper_bounds - self.lower_bounds)
        new_solution = solution + perturbation
        new_solution = np.clip(new_solution, self.lower_bounds, self.upper_bounds)
        self.perturbation_strength *= 1.1 # Increase perturbation strength adaptively
        return new_solution

    def _is_stagnant(self, solution):
        return np.allclose(solution, self.best_solution_overall, atol=1e-4)


    def _update_tabu_list(self, solution):
        self.tabu_list.append(tuple(solution))
        if len(self.tabu_list) > self.tabu_length:
            self.tabu_list.pop(0)

    def _fitness(self, solution, objective_function):
        return objective_function(solution.reshape(1, -1))[0]

    def _accept(self, new_fitness, current_fitness, temperature):
        if new_fitness < current_fitness:
            return True
        else:
            delta_e = new_fitness - current_fitness
            acceptance_probability = np.exp(-delta_e / temperature)
            return random.random() < acceptance_probability







