import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Callable, Dict, Optional, Union
from scipy.io import loadmat
import os
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging
from dataclasses import dataclass
from scipy.stats import rankdata
import warnings
# --- Parameter Dataclass ---
@dataclass
class ERADSParams:
    """Configuration parameters for ERADS_QuantumFluxUltraRefined"""
    population_size: int = 50
    max_generations: int = 5000
    F_init: float = 0.55  # Initial scaling factor for mutation
    F_end: float = 0.85  # Final scaling factor for mutation
    CR: float = 0.95  # Crossover probability
    memory_factor: float = 0.3  # Memory factor for mutation guidance


# --- Refactored Algorithm ---
class ERADS_QuantumFluxUltraRefined_GNBG:
    def __init__(
        self,
        problem_dim: int,
        bounds: Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], # (lower_bounds, upper_bounds)
        params: Optional[ERADSParams] = None,
    ):
        self.dim = problem_dim
        self.lb, self.ub = bounds  # Store lower and upper bounds

        # Use provided params or default
        self.params = params if params is not None else ERADSParams()

        self.x_opt = None
        self.f_opt = np.inf

    def optimize(
        self,
        fitness_func: Callable[[np.ndarray], float],
        max_evals: int,
        acceptance_threshold: float = 1e-8 # Matches GNBG example, not used for stopping in this version
    ) -> Tuple[Optional[np.ndarray], float, Dict]:
        
        # Initialize population uniformly within the bounds
        # np.random.uniform handles scalar or array bounds correctly
        population = np.random.uniform(
            self.lb, self.ub, (self.params.population_size, self.dim)
        )

        # Evaluate initial population
        fitness = np.zeros(self.params.population_size)
        evaluations = 0
        for i in range(self.params.population_size):
            if evaluations < max_evals:
                # GNBG expects input as 2D array for each individual
                solution_2d = population[i].reshape(1, -1)
                fit_val = fitness_func(solution_2d)
                # Ensure fitness is a scalar
                fitness[i] = fit_val[0] if isinstance(fit_val, np.ndarray) else fit_val
                evaluations += 1
            else:
                fitness[i] = np.inf # Or handle gracefully if budget runs out during init

        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index].copy()
        
        # Initialize memory for successful mutation directions
        memory = np.zeros(self.dim)

        while evaluations < max_evals:
            # Linear adaptation of the scaling factor F
            F_current = self.params.F_init + (self.params.F_end - self.params.F_init) * (
                evaluations / max_evals
            )

            for i in range(self.params.population_size):
                if evaluations >= max_evals:
                    break

                # Selection of three distinct random population indices
                indices = np.random.choice(
                    [j for j in range(self.params.population_size) if j != i], 3, replace=False
                )
                x1, x2, x3 = population[indices]
                
                # Use current global best for mutation
                current_best_sol = self.x_opt 

                # Mutant vector creation (ERADS specific strategy)
                mutant = x1 + F_current * (
                    current_best_sol - x1 + x2 - x3 + self.params.memory_factor * memory
                )
                # Ensure mutant remains within bounds (self.lb and self.ub can be scalar or array)
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                trial_mask = np.random.rand(self.dim) < self.params.CR
                # Ensure at least one gene from mutant
                if not np.any(trial_mask):
                    trial_mask[np.random.randint(0, self.dim)] = True
                trial = np.where(trial_mask, mutant, population[i])
                
                # Evaluate trial vector
                solution_2d = trial.reshape(1, -1)
                f_trial_val = fitness_func(solution_2d)
                f_trial = f_trial_val[0] if isinstance(f_trial_val, np.ndarray) else f_trial_val
                evaluations += 1

                # Selection
                if f_trial < fitness[i]:
                    # Update memory with the scaled successful mutation direction component
                    # Note: The original logic `mutant - population[i]` might refer to the old population[i]
                    # If population[i] is updated first, this difference changes.
                    # Assuming it's the difference leading to the improvement:
                    successful_step = trial - population[i] # Or mutant - population[i] if that's intended base for memory

                    population[i] = trial
                    fitness[i] = f_trial
                    
                    memory = (
                        1 - self.params.memory_factor
                    ) * memory + self.params.memory_factor * F_current * successful_step


                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()
                        # best_index = i # This is implicitly handled as self.x_opt is updated directly

                if evaluations >= max_evals:
                    break
        
        optimization_info = {
            "function_evaluations": evaluations,
            "final_F_current": F_current, # Example of what can be added
        }
        return self.x_opt, self.f_opt, optimization_info
    

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
        params = ERADSParams(
            population_size=min(100, 10 * gnbg.Dimension),
            max_generations=gnbg.MaxEvals // min(100, 10 * gnbg.Dimension)
        )
        
        optimizer = ERADS_QuantumFluxUltraRefined_GNBG(
            problem_dim=gnbg.Dimension,
            bounds=bounds,
            params=params
        )
        
        # Run optimization
        best_solution, best_fitness, _ = optimizer.optimize(
            fitness_func=gnbg.fitness,
            max_evals=gnbg.MaxEvals,
            acceptance_threshold=gnbg.AcceptanceThreshold
        )
        
        best_values.append(best_fitness)
        best_solutions.append(best_solution)
        
        logging.info(f"Run {run + 1} completed. Best fitness: {best_fitness:.6e}")
    
    return best_values, best_solutions

if __name__ == "__main__":
    folder_path = "codes/gnbg_python"
    # Example usage
    ProblemIndex = 1
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
    print(f"Best fitness values: {best_values}")
    print(f"Mean fitness: {np.mean(best_values):.6e}")
    print(f"Std fitness: {np.std(best_values):.6e}") 