task_prompt_gnbg = """
Your goal is to design a novel optimization algorithm for box-constrained numerical global optimization problems. This algorithm will be primarily evaluated 
on the GNBG (Generalized Numerical Benchmark Generator) test suite, which consists of 24 challenging functions.
The primary performance metric will be the Area Over the Convergence Curve (AOCC), where higher is better.

Your task is to write this optimization algorithm in Python code. The algorithm should be implemented as a Python class.

The class structure should include at least:
1.  An `__init__(self, budget, dim, lower_bounds, upper_bounds)` method:
    * `budget` (int): The maximum number of function evaluations allowed for a single optimization run.
    * `dim` (int): The dimensionality of the problem (typically 30 for GNBG).
    * `lower_bounds` (list or NumPy array): Lower bound for each variable.
    * `upper_bounds` (list or NumPy array): Upper bound for each variable.
    * This method should initialize any parameters or state your algorithm needs.

2.  An `optimize(self, objective_function)` method:
    * `objective_function` (callable): The black-box function to be minimized. It accepts a 2D NumPy array `X` (rows are solutions) and returns a 1D NumPy array of fitness values.
    * This method must implement your algorithm's logic to find the solution vector that minimizes the `objective_function`, strictly adhering to `self.budget`.
    * It should return two values:
        1.  The best solution vector (1D NumPy array) found.
        2.  The fitness value (scalar) of that best solution.

For the GNBG benchmark:
* `dim` is typically 30.
* Bounds are typically -100.0 to 100.0 for each dimension.

Please provide an excellent and novel optimization algorithm.
Structure your response as follows:
# Name: YourAlgorithmName
# Description: A concise one-line description of the main idea.
# Code:
```python
# You may import other standard Python libraries like 'random' or 'math' if needed.
# Avoid non-standard library imports.

class YourUniqueAlgorithmName:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        # Initialize best_solution_overall with a random valid point or None
        if self.dim > 0 and self.lower_bounds.size == self.dim and self.upper_bounds.size == self.dim:
            self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
        else:
            self.best_solution_overall = np.zeros(self.dim) if self.dim > 0 else np.array([])
        self.best_fitness_overall = float('inf')
        
        # LLM: Initialize any other essential algorithm-specific parameters or state here.
        # For example, a population if your algorithm is population-based:
        # self.population_size = 50 # Or some function of dim/budget
        # self.population = None # To be initialized in optimize() or here

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        # Reset or confirm state for this optimization run
        self.eval_count = 0
        if self.dim > 0 and self.lower_bounds.size == self.dim and self.upper_bounds.size == self.dim:
            self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
        else:
            self.best_solution_overall = np.zeros(self.dim) if self.dim > 0 else np.array([])
        self.best_fitness_overall = float('inf')
        
        # --- LLM: Implement your novel optimization algorithm logic below ---
        # This is a minimal placeholder showing interaction. Your algorithm will be complex.
        # Remember to manage self.eval_count carefully.

        # Example: Evaluate a single random candidate (LLM should do much more!)
        if self.eval_count < self.budget and self.dim > 0:
            # 1. Generate candidate solution(s) - must be a 2D NumPy array
            candidate_batch = np.random.uniform(self.lower_bounds, self.upper_bounds, (1, self.dim))
            
            # 2. Call objective function (respecting budget for this batch call)
            num_in_batch = candidate_batch.shape[0]
            evals_possible_now = self.budget - self.eval_count
            
            if evals_possible_now >= num_in_batch:
                try:
                    fitness_values = objective_function(candidate_batch) # Expects (N, dim), returns (N,)
                    self.eval_count += num_in_batch

                    # 3. Update best solution found so far
                    if fitness_values[0] < self.best_fitness_overall:
                        self.best_fitness_overall = fitness_values[0]
                        self.best_solution_overall = candidate_batch[0, :].copy()
                except Exception as e:
                    # print(f"Warning: Objective function call failed: {e}")
                    pass # Fitness will remain inf for this attempt
            # else: not enough budget for this evaluation
        
        # Your main algorithm loop should go here, e.g.:
        # while self.eval_count < self.budget:
        #     if self.best_fitness_overall <= acceptance_threshold:
        #         break
        #     # ... your EA/swarm/etc. logic ...
        #     # ... generate new solutions ...
        #     # ... evaluate them using objective_function, update self.eval_count ...
        #     # ... update self.best_solution_overall, self.best_fitness_overall ...
        # --- End of LLM's Core Algorithm Logic ---
        
        # Prepare diagnostic info
        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
            # LLM: You can add more algorithm-specific diagnostic data here
        }

        return self.best_solution_overall, self.best_fitness_overall, optimization_info
```
# Example of how the class might be used (for the LLM's understanding):
# bounds_example = [(-100.0, 100.0)] * 30
# optimizer = YourAlgorithmName(budget=500000, dim=30, lower_bounds=[b[0] for b in bounds_example], upper_bounds=[b[1] for b in bounds_example])
# def dummy_objective_function(X_batch): # X_batch is (N,D)
#     return np.sum(X_batch**2, axis=1) # Returns (N,)
# best_s, best_f = optimizer.optimize(dummy_objective_function)
"""
simplified_task_prompt = '''
Your objective is to design a novel Python optimization algorithm class for box-constrained numerical global optimization, specifically for the GNBG benchmark (24 functions, typically 30 dimensions, bounds typically -100.0 to 100.0). The algorithm should aim to MINIMIZE the objective function value. Performance will be assessed using AOCC (Area Over the Convergence Curve), where higher AOCC is better.

The Python class **must** implement:

1.  `__init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float])`:
    * `budget`: Max function evaluations. Store as `self.budget`.
    * `dim`: Problem dimensionality. Store as `self.dim`.
    * `lower_bounds`, `upper_bounds`: Lists of floats for variable boundaries. Store as 1D NumPy arrays `self.lower_bounds` and `self.upper_bounds`.
    * Initialize `self.eval_count = 0`, `self.best_solution_overall = None`, `self.best_fitness_overall = float('inf')`.

2.  `optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple`:
    * `objective_function`: Accepts a 2D NumPy array `X` (shape `(N, self.dim)`) and returns a 1D NumPy array of `N` fitness values.
    * Implement your algorithm's core logic here.
    * Strictly manage `self.eval_count` so it does not exceed `self.budget` when calling `objective_function`.
    * Ensure solutions generated and returned respect `self.lower_bounds` and `self.upper_bounds`.
    * Return a tuple: `(best_solution_1D_numpy_array, best_fitness_scalar, optimization_info_dict)`.
        * `optimization_info_dict` should at least contain `{'function_evaluations_used': self.eval_count, 'final_best_fitness': self.best_fitness_overall}`.

**Output Format:**

# Name: YourUniqueAlgorithmName
# Description: Concise one-line description of the algorithm's main idea.
# Code:
```python
import numpy as np
# Add other standard library imports if needed (e.g., random).

class YourUniqueAlgorithmName:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None # Will be a 1D NumPy array
        self.best_fitness_overall = float('inf')
        
        # LLM: Initialize any algorithm-specific state here

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0 # Reset for this run
        # Initialize best_solution_overall with a valid random point or leave as None
        if self.dim > 0:
             self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
        else:
             self.best_solution_overall = np.array([])
        self.best_fitness_overall = float('inf')

        # --- LLM: Implement core optimization logic here ---
        # Example:
        # while self.eval_count < self.budget:
        #     # 1. Generate candidate_batch (2D NumPy array)
        #     # 2. Ensure candidate_batch is within bounds
        #     # 3. Evaluate:
        #     #    num_to_eval_now = min(candidate_batch.shape[0], self.budget - self.eval_count)
        #     #    if num_to_eval_now <= 0: break
        #     #    actual_batch = candidate_batch[:num_to_eval_now,:]
        #     #    fitness_values = objective_function(actual_batch)
        #     #    self.eval_count += actual_batch.shape[0]
        #     #    # Update self.best_solution_overall & self.best_fitness_overall
        #     # 4. Check acceptance_threshold
        #     pass # Replace with actual logic
        # --- End LLM Logic ---

        if self.best_solution_overall is None and self.dim > 0 : # Fallback if no evaluations made
            self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
            
        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info
'''