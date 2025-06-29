from scipy.io import loadmat
import os
import logging
from typing import Optional, Tuple, List
from my_utils.utils import calculate_aocc_from_gnbg_history
import numpy as np
import random
from codes.gnbg_python.GNBG_instances import GNBG
from datetime import datetime
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

folder = "log_test_algorithms"
os.makedirs(folder, exist_ok=True)

# Unique log filename based on timestamp
log_filename = datetime.now().strftime(f"{folder}/run_%Y%m%d_%H%M%S.log")

# Configure logging
logging.basicConfig(
    filename=log_filename,
    filemode='w',  # overwrite if exists
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


import numpy as np
import random
from scipy.optimize import minimize

import numpy as np

import numpy as np
import random

# Name: AdaptiveIslandDE
# Description: An island-based DE with adaptive migration, local search, and restart mechanisms to handle deceptive landscapes.
# Code:
# Terrible at: f9, f24, f13, f14, f15, f21, f22
import numpy as np
import random
import logging

class AdaptiveIslandDE:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float],
                 num_islands: int = 5, population_size: int = 20, crossover_rate: float = 0.8719572569354708,
                 mutation_rate: float = 0.6113964692124271, migration_interval: int = 896.9508697672186, migration_size: int = 2.414743986796276,
                 local_search_iterations: int = 5.957280686848644, local_search_perturbation_scale: float = 0.1446330223199665, restart_percentage: float = 0.7,stagnation_threshold: float = 1e-2, stagnation_patient: int = 200, adaptive_stagnation: bool = True, initial_stagnation_threshold: float = 1e-2, final_stagnation_threshold: float = 1e-8):
        """
        Initializes the AdaptiveIslandDE optimizer.

        Args:
            budget (int): Maximum number of function evaluations.
            dim (int): Problem dimensionality.
            lower_bounds (list[float]): List of lower bounds for each dimension.
            upper_bounds (list[float]): List of upper bounds for each dimension.
            num_islands (int): Number of independent subpopulations (islands).
            population_size (int): Number of individuals in each island's population.
            crossover_rate (float): DE crossover probability.
            mutation_rate (float): DE mutation scaling factor.
            migration_interval (int): Number of evaluations between migrations.
            migration_size (int): Number of individuals to migrate.
            local_search_iterations (int): Iterations for local search
            local_search_perturbation_scale (float): Scale of the random perturbation in local search.
            restart_percentage (float): Percentage of budget used to trigger restart.
        """
        self.initial_stagnation_threshold = initial_stagnation_threshold
        self.final_stagnation_threshold = final_stagnation_threshold
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)
        self.num_islands = num_islands
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.migration_interval = int(migration_interval)
        self.migration_size = int(migration_size)
        self.local_search_iterations = int(local_search_iterations)
        self.local_search_perturbation_scale = local_search_perturbation_scale
        self.restart_percentage = restart_percentage
        self.adaptive_stagnation = adaptive_stagnation
        # Restart parameters
        self.stagnation_patience = stagnation_patient
        self.stagnation_threshold = stagnation_threshold
        self.fitness_history = [[] for _ in range(num_islands)]
        
        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')

        # Initialize populations for each island
        self.populations = [
            np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
            for _ in range(self.num_islands)
        ]
        self.fitness_values = [np.full(self.population_size, float('inf')) for _ in range(self.num_islands)]
        self.best_solutions = [None] * self.num_islands
        self.best_fitnesses = [float('inf')] * self.num_islands


    def differential_evolution_step(self, island_index: int, objective_function: callable):
        """
        Performs a single step of differential evolution on a given island.

        Args:
            island_index (int): Index of the island to evolve.
            objective_function (callable): The objective function to optimize.
        """

        population = self.populations[island_index]
        fitness_values = self.fitness_values[island_index]

        for i in range(self.population_size):
            # Mutation
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = random.sample(indices, 3)
            mutant_vector = population[a] + self.mutation_rate * (population[b] - population[c])
            mutant_vector = np.clip(mutant_vector, self.lower_bounds, self.upper_bounds)  # Clip to bounds

            # Crossover
            trial_vector = np.copy(population[i])
            for j in range(self.dim):
                if random.random() < self.crossover_rate:
                    trial_vector[j] = mutant_vector[j]

            # Evaluation
            trial_vector_reshaped = trial_vector.reshape(1, -1)
            trial_fitness = objective_function(trial_vector_reshaped)[0]
            self.eval_count += 1

            # Selection
            if trial_fitness < fitness_values[i]:
                population[i] = trial_vector
                fitness_values[i] = trial_fitness

                # Update island best
                if trial_fitness < self.best_fitnesses[island_index]:
                    self.best_fitnesses[island_index] = trial_fitness
                    self.best_solutions[island_index] = trial_vector

                # Update overall best
                if trial_fitness < self.best_fitness_overall:
                    self.best_fitness_overall = trial_fitness
                    self.best_solution_overall = trial_vector

        self.populations[island_index] = population
        self.fitness_values[island_index] = fitness_values

    def local_search(self, solution: np.ndarray, objective_function: callable) -> tuple:
        """
        Performs local search around a solution using a simple gradient-based method.

        Args:
            solution (np.ndarray): The solution to start the local search from.
            objective_function (callable): The objective function to optimize.

        Returns:
            tuple: A tuple containing the improved solution and its fitness.
        """
        best_solution = solution.copy()
        best_fitness = objective_function(best_solution.reshape(1, -1))[0]
        self.eval_count += 1

        for _ in range(self.local_search_iterations):
            # Create a small random perturbation
            perturbation = np.random.normal(0, self.local_search_perturbation_scale, self.dim) # Scale adjusted from 1 to 0.1
            new_solution = best_solution + perturbation
            new_solution = np.clip(new_solution, self.lower_bounds, self.upper_bounds)

            # Evaluate the new solution
            new_fitness = objective_function(new_solution.reshape(1, -1))[0]
            self.eval_count += 1

            # If the new solution is better, update the current best
            if new_fitness < best_fitness:
                best_fitness = new_fitness
                best_solution = new_solution

        return best_solution, best_fitness


    def migrate(self, objective_function: callable):
        """
        Migrates individuals between islands.  Each island sends its best
        individuals to a randomly chosen other island, and receives
        migrants to replace its worst individuals.
        """

        for i in range(self.num_islands):
            # Select a random destination island (excluding itself)
            dest_island = random.choice([j for j in range(self.num_islands) if j != i])

            # Identify the best solutions on the source island
            source_island_fitness = self.fitness_values[i]
            best_indices = np.argsort(source_island_fitness)[:self.migration_size]
            migrants = self.populations[i][best_indices].copy()  # Important to copy

            # Identify the worst solutions on the destination island
            dest_island_fitness = self.fitness_values[dest_island]
            worst_indices = np.argsort(dest_island_fitness)[-self.migration_size:]

            # Replace the worst solutions on the destination island with the migrants
            self.populations[dest_island][worst_indices] = migrants

            # Re-evaluate the fitness of the new solutions on the destination island (important!) and perform local search
            new_fitnesses = []
            for j in range(len(worst_indices)):
                migrant = migrants[j]
                migrant, fitness = self.local_search(migrant, objective_function) #Local Adaptation here.
                new_fitnesses.append(fitness)

            dest_island_fitness[worst_indices] = new_fitnesses
            self.fitness_values[dest_island] = dest_island_fitness
             # Update best fitness, if needed
            for fit, sol in zip(new_fitnesses, migrants):
                if fit < self.best_fitnesses[dest_island]:
                     self.best_fitnesses[dest_island] = fit
                     self.best_solutions[dest_island] = sol
                if fit < self.best_fitness_overall:
                    self.best_fitness_overall = fit
                    self.best_solution_overall = sol


    def detect_stagnation(self, fitness_history: list[float]) -> bool:
        if len(fitness_history) < self.stagnation_patience:
            return False
        recent = fitness_history[-self.stagnation_patience:]
    
        if self.adaptive_stagnation:
            # Log-scale interpolation
            log_start = np.log10(self.initial_stagnation_threshold)
            log_end = np.log10(self.final_stagnation_threshold)
            progress = self.eval_count / self.budget
            adaptive_threshold = 10 ** (log_start + progress * (log_end - log_start))
        else:
            adaptive_threshold = self.initial_stagnation_threshold

        std_recent = np.std(recent)
        
        # Optional debug
        logging.debug(f"FE={self.eval_count}, std={std_recent:.2e}, adaptive threshold={adaptive_threshold:.2e}")
        
        return std_recent < adaptive_threshold


    def restart_island(self, i: int, objective_function: callable):
        logging.info(f"Restarting island {i} due to stagnation at FE={self.eval_count}")
        self.populations[i] = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
        self.fitness_values[i] = objective_function(self.populations[i])
        self.eval_count += self.population_size
        best_index = np.argmin(self.fitness_values[i])
        self.best_fitnesses[i] = self.fitness_values[i][best_index]
        self.best_solutions[i] = self.populations[i][best_index]

        if self.best_fitnesses[i] < self.best_fitness_overall:
            self.best_fitness_overall = self.best_fitnesses[i]
            self.best_solution_overall = self.best_solutions[i]

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8, optimum_value=None) -> tuple:
        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')

        # Initialize fitness values for each island
        for i in range(self.num_islands):
            self.fitness_values[i] = objective_function(self.populations[i])
            self.eval_count += self.population_size
            best_index = np.argmin(self.fitness_values[i])
            self.best_fitnesses[i] = self.fitness_values[i][best_index]
            self.best_solutions[i] = self.populations[i][best_index]
            self.fitness_history[i].append(self.best_fitnesses[i])

            if self.best_fitnesses[i] < self.best_fitness_overall:
                self.best_fitness_overall = self.best_fitnesses[i]
                self.best_solution_overall = self.best_solutions[i]

        # Main optimization loop
        while self.eval_count < self.budget:
            if optimum_value is not None and abs(self.best_fitness_overall - optimum_value) <= acceptance_threshold:
                logging.info(f"Stopping early: Acceptance threshold {acceptance_threshold} reached at FE {self.eval_count}.")
                break

            for i in range(self.num_islands):
                self.differential_evolution_step(i, objective_function)
                self.fitness_history[i].append(self.best_fitnesses[i])

            if self.eval_count % self.migration_interval == 0:
                self.migrate(objective_function)

            # Restart islands based on stagnation detection
            for i in range(self.num_islands):
                if self.detect_stagnation(self.fitness_history[i]):
                    self.restart_island(i, objective_function)

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall,
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info





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
    base_seed = 42
    problem_dir = f"results/problem_{ProblemIndex}"
    
    # Load problem instance
    from codes.gnbg_python.GNBG_instances import GNBG
    # Set up bounds
    
    # Initialize results storage
    best_values = []
    best_solutions = []
    aoccs = []
    for run in range(num_runs):
        run_seed = base_seed + run  # Vary seed per run
        np.random.seed(run_seed)
        random.seed(run_seed)
        gnbg = GNBG(MaxEvals, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate, CompMinPos, CompSigma, CompH, Mu, Omega, Lambda, RotationMatrix, OptimumValue, OptimumPosition)
        bounds = (gnbg.MinCoordinate, gnbg.MaxCoordinate)
        logging.info(f"Starting run {run + 1}/{num_runs}")
        
        # Initialize algorithm
       
        try:
            optimizer = AdaptiveIslandDE(
                budget=MaxEvals,
                dim=gnbg.Dimension,
                lower_bounds=[gnbg.MinCoordinate for _ in range(gnbg.Dimension)],
                upper_bounds=[gnbg.MaxCoordinate for _ in range(gnbg.Dimension)],
            )
            
            # Run optimization
            best_solution, best_fitness, _ = optimizer.optimize(
                objective_function=gnbg.fitness, optimum_value = OptimumValue
            )
            
            aocc = calculate_aocc_from_gnbg_history(
                fe_history=gnbg.FEhistory,
                optimum_value=gnbg.OptimumValue,
                budget_B=gnbg.MaxEvals
            )
            best_values.append(best_fitness)
            best_solutions.append(best_solution)
            aoccs.append(aocc)
            
            logging.info(f"Run {run + 1} completed. Best fitness: {best_fitness:.6e}, AOCC: {aocc:.4f}")
            logging.info(f"\nResults for Problem {ProblemIndex}:")
            logging.info(f"Best solution: {best_solutions}")
            logging.info(f"Optimun Solution: {OptimumValue}")
            logging.info(f"Best fitness values: {best_values}")
            logging.info(f"Mean fitness: {np.mean(best_values)}")
            logging.info(f"Std fitness: {np.std(best_values)}") 
            logging.info(f"Mean AOCC:         {np.mean(aoccs):.4f} (Higher is better)")
            logging.info(f"Std Dev AOCC:      {np.std(aoccs):.4f}")
            
        except Exception as e:
            logging.error(f"Run {run + 1} failed due to: {e}", exc_info=True)
            print(f"Run {run + 1} failed: {e}")
        
        convergence = []
        best_error = float('inf')
        for value in gnbg.FEhistory:
            error = abs(value - OptimumValue)
            if error < best_error:
                best_error = error
            convergence.append(best_error)

        # Plotting the convergence
        plt.plot(range(1, len(convergence) + 1), convergence)
        plt.xlabel('Function Evaluation Number (FE)')
        plt.ylabel('Error')
        plt.title('Convergence Plot')
        plt.yscale('log')  # Set y-axis to logarithmic scale  
        plt.ylim(bottom=1e-8)  # Set lower limit to 10^-8
        plt.tight_layout()
        plt.savefig(f"{folder}/convergence_problem{ProblemIndex}_run{run + 1}.png")
        plt.close()
        
if __name__ == "__main__":
    folder_path = "codes/gnbg_python"
    solutions_dir = os.path.join(folder, "best_solutions")
    os.makedirs(solutions_dir, exist_ok=True)
    # Example usage
    problem_list = [1,2,3,4,5,6,7,8,9,10]
    for ProblemIndex in problem_list:
        
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
        
        run_optimization(700000, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate,
                                                       CompMinPos, CompSigma, CompH, Mu, Omega, Lambda, RotationMatrix, OptimumValue, OptimumPosition)
    
    
    # print(f"\nResults for Problem {ProblemIndex}:")
    # print(f"Best solution: {best_solutions}")
    # print(f"Optimun Solution: {OptimumValue}")
    # print(f"Best fitness values: {best_values[1:]}")
    # print(f"Mean fitness: {np.mean(best_values[1:])}")
    # print(f"Std fitness: {np.std(best_values[1:])}") 
    # print(f"Mean AOCC:         {np.mean(aoccs):.4f} (Higher is better)")
    # print(f"Std Dev AOCC:      {np.std(aoccs):.4f}")
    
# python test_generated_algorithm.py