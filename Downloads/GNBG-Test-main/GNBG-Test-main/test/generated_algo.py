import numpy as np
from scipy.io import loadmat
import os
import logging
from typing import Optional, Tuple, List
from test.calc_aocc_from_gnbg import calculate_aocc_from_gnbg_history
class HybridPSO_NelderMead:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
        self.best_fitness_overall = float('inf')

        self.population_size = 50
        self.population = None
        self.velocities = None
        self.personal_bests = None
        self.personal_best_fitness = None
        self.global_best = None
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.4
        self.social_coefficient = 1.4


    def optimize(self, objective_function: callable) -> tuple:
        self.eval_count = 0
        self.population = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
        self.velocities = np.zeros_like(self.population)
        self.personal_bests = self.population.copy()
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best = None

        fitness_values = objective_function(self.population)
        self.eval_count += self.population_size
        
        for i, fitness in enumerate(fitness_values):
            if fitness < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness
                self.personal_bests[i] = self.population[i].copy()
            if fitness < self.best_fitness_overall:
                self.best_fitness_overall = fitness
                self.best_solution_overall = self.population[i].copy()
                self.global_best = self.population[i].copy()

        while self.eval_count < self.budget:
            self.update_velocities()
            self.update_positions()
            fitness_values = objective_function(self.population)
            self.eval_count += self.population_size
            for i, fitness in enumerate(fitness_values):
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_bests[i] = self.population[i].copy()
                if fitness < self.best_fitness_overall:
                    self.best_fitness_overall = fitness
                    self.best_solution_overall = self.population[i].copy()
                    self.global_best = self.population[i].copy()

            #Nelder-Mead local search
            self.nelder_mead_local_search(objective_function)


        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }

        return self.best_solution_overall, self.best_fitness_overall, optimization_info


    def update_velocities(self):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)
        self.velocities = self.inertia_weight * self.velocities + \
                          self.cognitive_coefficient * r1 * (self.personal_bests - self.population) + \
                          self.social_coefficient * r2 * (self.global_best - self.population)

    def update_positions(self):
        self.population = self.population + self.velocities
        self.population = np.clip(self.population, self.lower_bounds, self.upper_bounds)


    def nelder_mead_local_search(self, objective_function):
        if self.eval_count < self.budget:
            simplex = np.stack([self.best_solution_overall] + [np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim) for _ in range(self.dim)])
            
            for _ in range(100): # Limit iterations to avoid excessive local search time
                if self.eval_count >= self.budget:
                    break
                fitness_values = objective_function(simplex)
                self.eval_count += simplex.shape[0]
                #Simple Nelder Mead implementation, can be improved
                sorted_indices = np.argsort(fitness_values)
                best_point = simplex[sorted_indices[0]]
                worst_point = simplex[sorted_indices[-1]]
                centroid = np.mean(simplex[sorted_indices[:-1]], axis=0)
                reflection_point = 2 * centroid - worst_point
                reflection_fitness = objective_function(reflection_point.reshape(1, -1))
                self.eval_count += 1

                if reflection_fitness < self.best_fitness_overall:
                    self.best_fitness_overall = reflection_fitness
                    self.best_solution_overall = reflection_point.copy()
                    simplex[sorted_indices[-1]] = reflection_point
                else:
                    break # Simplified termination condition
def run_optimization(MaxEvals, AcceptanceThreshold, 
                     Dimension, CompNum, MinCoordinate, MaxCoordinate,
                     CompMinPos, CompSigma, CompH, Mu, Omega,
                     Lambda, RotationMatrix, OptimumValue, OptimumPosition,
                    num_runs: int = 31,
                    seed: Optional[int] = None) -> Tuple[List[float], List[np.ndarray]]:
    """
    Run multiple optimization runs for a given problem
    
    Args:
        problem_index: GNBG problem index (1-24)
        num_runs: Number of independent runs
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (best_fitness_values, best_solutions)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Load problem instance
    from codes.gnbg_python.GNBG_instances import GNBG
    gnbg = GNBG(MaxEvals, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate, CompMinPos, CompSigma, CompH, Mu, Omega, Lambda, RotationMatrix, OptimumValue, OptimumPosition)
    # Set up bounds
    bounds = (gnbg.MinCoordinate, gnbg.MaxCoordinate)
    
    # Initialize results storage
    best_values = []
    best_solutions = []
    
    for run in range(num_runs):
        logging.info(f"Starting run {run + 1}/{num_runs}")
        
        # Initialize algorithm
       
        
        optimizer = HybridPSO_NelderMead(
            budget=gnbg.MaxEvals,
            dim=gnbg.Dimension,
            lower_bounds=[gnbg.MinCoordinate for _ in range(gnbg.Dimension)],
            upper_bounds=[gnbg.MaxCoordinate for _ in range(gnbg.Dimension)]
        )
        
        # Run optimization
        best_solution, best_fitness, _ = optimizer.optimize(
            objective_function=gnbg.fitness
        )
        
        best_values.append(best_fitness)
        best_solutions.append(best_solution)
        
        logging.info(f"Run {run + 1} completed. Best fitness: {best_fitness}")
    
    return best_values, best_solutions
if __name__ == "__main__":
    folder_path = "codes/gnbg_python"
    # Example usage
    ProblemIndex = 2
    if 1 <= ProblemIndex <= 24:
        
        filename = f'f{ProblemIndex}.mat'
        GNBG_tmp = loadmat(os.path.join(folder_path, filename))['GNBG']
        MaxEvals = np.array([item[0] for item in GNBG_tmp['MaxEvals'].flatten()])[0, 0]
        AcceptanceThreshold = np.array([item[0] for item in GNBG_tmp['AcceptanceThreshold'].flatten()])[0, 0]
        Dimension = np.array([item[0] for item in GNBG_tmp['Dimension'].flatten()])[0, 0]
        CompNum = np.array([item[0] for item in GNBG_tmp['o'].flatten()])[0, 0]  # Number of components
        MinCoordinate = np.array([item[0] for item in GNBG_tmp['MinCoordinate'].flatten()])[0, 0]
        MaxCoordinate = np.array([item[0] for item in GNBG_tmp['MaxCoordinate'].flatten()])[0, 0]
        CompMinPos = np.array(GNBG_tmp['Component_MinimumPosition'][0, 0])
        CompSigma = np.array(GNBG_tmp['ComponentSigma'][0, 0], dtype=np.float64)
        CompH = np.array(GNBG_tmp['Component_H'][0, 0])
        Mu = np.array(GNBG_tmp['Mu'][0, 0])
        Omega = np.array(GNBG_tmp['Omega'][0, 0])
        Lambda = np.array(GNBG_tmp['lambda'][0, 0])
        RotationMatrix = np.array(GNBG_tmp['RotationMatrix'][0, 0])
        OptimumValue = np.array([item[0] for item in GNBG_tmp['OptimumValue'].flatten()])[0, 0]
        OptimumPosition = np.array(GNBG_tmp['OptimumPosition'][0, 0])
        
        best_values, best_solutions = run_optimization(MaxEvals, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate,
                                                       CompMinPos, CompSigma, CompH, Mu, Omega, Lambda, RotationMatrix, OptimumValue, OptimumPosition)
    else:
        raise ValueError('ProblemIndex must be between 1 and 24.')
    
    print(f"\nResults for Problem {ProblemIndex}:")
    print(f"Best solution: {best_solutions}")
    
    print(f"Best fitness values: {best_values[1:]}")
    print(f"Mean fitness: {np.mean(best_values[1:])}")
    print(f"Std fitness: {np.std(best_values[1:])}") 