import os
import numpy as np
from scipy.io import loadmat
import random
import traceback
import contextlib # For redirect_stdout
import json # For Solution class
import uuid  # For Solution class
from LLaMEA.llamea import Solution
from codes.gnbg_python.GNBG_instances import GNBG
from LLaMEA.llamea import LLaMEA, Gemini_LLM, Ollama_LLM


# --- Helper Function to Calculate AOCC for GNBG ---
def calculate_aocc_from_gnbg_history(fe_history, optimum_value, budget_B, 
                                     log_error_lower_bound=-8.0,  # Corresponds to 10^-8 error
                                     log_error_upper_bound=2.0):   # Corresponds to 10^2 error
    """
    Calculates Area Over the Convergence Curve (AOCC) from GNBG FEhistory.
    Higher AOCC is better (1.0 is optimal).
    """
    if not fe_history:
        return 0.0 # No evaluations, worst AOCC

    actual_evals = len(fe_history)
    
    best_error_so_far = float('inf')
    best_error_history_at_fe = []

    for i in range(actual_evals):
        current_fitness = fe_history[i]
        if np.isnan(current_fitness) or np.isinf(current_fitness): # Handle invalid fitness from GNBG
            current_error = float('inf')
        else:
            current_error = abs(current_fitness - optimum_value)
        
        best_error_so_far = min(best_error_so_far, current_error)
        best_error_history_at_fe.append(best_error_so_far)

    # If fe_history is shorter than budget_B, extend with the last best error
    if actual_evals < budget_B:
        best_error_history_at_fe.extend([best_error_so_far] * (budget_B - actual_evals))
    
    aocc_terms = []
    for error_at_fe in best_error_history_at_fe[:budget_B]: # Ensure we only consider up to budget_B
        # Floor error at a very small positive number to avoid log(0) or log(<0)
        log_error = np.log10(max(error_at_fe, 1e-100)) 
        
        clipped_log_error = np.clip(log_error, log_error_lower_bound, log_error_upper_bound)
        
        # Normalize to [0, 1], where 0 is best error (at lower_bound), 1 is worst (at upper_bound)
        range_log_error = log_error_upper_bound - log_error_lower_bound
        if range_log_error <= 0: # Avoid division by zero if bounds are weird
            normalized_value = 0.0 if log_error <= log_error_lower_bound else 1.0
        else:
            normalized_value = (clipped_log_error - log_error_lower_bound) / range_log_error
        
        score_at_fe = 1.0 - normalized_value # Higher score for lower normalized error
        aocc_terms.append(score_at_fe)
        
    return np.mean(aocc_terms) if aocc_terms else 0.0


# --- Evaluation Function for GNBG Benchmark ---
def evaluateGNBG(solution, explogger=None, details=False,
                 gnbg_problem_folder=".",
                 runs_per_gnbg_func_meta_eval=3): # Fewer runs for LLaMEA's internal eval
    """
    Evaluates an LLM-generated algorithm on the GNBG benchmark.
    Returns mean AOCC (higher is better for LLaMEA if minimization=False).
    """
    all_run_aocc_scores = []
    
    unimodal_aoccs = []       # f1-f6
    multimodal_single_aoccs = [] # f7-f15
    multimodal_multi_aoccs = []  # f16-f24

    code_to_execute = solution.code
    algorithm_name = solution.name
    
    execution_scope = {}
    try:
        exec(code_to_execute, execution_scope)
        AlgorithmClass = execution_scope.get(algorithm_name)
        if AlgorithmClass is None:
            raise NameError(f"Algorithm class '{algorithm_name}' not found after executing code.")
    except Exception as e:
        error_str = f"Error executing LLM-generated code for '{algorithm_name}': {e}\n{traceback.format_exc()}"
        print(error_str)
        # LLaMEA default is maximization, so lower fitness is worse.
        # If LLaMEA is set to minimization=True, it will try to minimize this score.
        # For AOCC (higher is better), a score of 0 or -inf is worst.
        solution.set_scores(0.0, "Execution error in generated code.", error_str) 
        return solution

    for problem_idx in range(1, 25): # f1 to f24
        try:
            filename = f'f{problem_idx}.mat'
            gnbg_data_path = os.path.join(gnbg_problem_folder, filename)
            if not os.path.exists(gnbg_data_path):
                print(f"Warning: .mat file not found: {gnbg_data_path}. Skipping f{problem_idx}.")
                continue
            
            gnbg_tmp_data = loadmat(gnbg_data_path)['GNBG']

            def get_param(data_dict, key, is_scalar=True, dtype=None):
                val_array = data_dict[key].flatten()
                if not val_array.size: return None
                val = val_array[0]
                if is_scalar and isinstance(val, np.ndarray) and val.ndim >=0 : # Handle 0-dim arrays from MATLAB
                    # Check if it's a scalar within a 1x1 array or just a scalar array
                    if val.size == 1:
                         val = val.item() # Get Python scalar
                    # else it's an array meant to be returned as such, but flatten took [0]
                    # This part needs to be careful based on .mat structure.
                    # For GNBG, most scalars are wrapped like [[value]].
                    elif val.ndim == 2 and val.shape[0] ==1 and val.shape[1] == 1:
                         val = val[0,0]

                if isinstance(val, np.ndarray) and not is_scalar : # if it's meant to be an array
                     pass # val is already an array
                elif is_scalar and not isinstance(val, np.ndarray): # it's already a python scalar
                     pass

                if dtype: return np.array(val, dtype=dtype) # Ensure it's np.array if not scalar
                return np.array(val) if not isinstance(val, (int, float, np.number)) else val


            problem_max_fevals = int(get_param(gnbg_tmp_data, 'MaxEvals')) # 50000
            problem_accept_thresh = float(get_param(gnbg_tmp_data, 'AcceptanceThreshold')) # 1e-08
            problem_dimension = int(get_param(gnbg_tmp_data, 'Dimension')) # 30 vector input
            problem_comp_num = int(get_param(gnbg_tmp_data, 'o'))
            problem_min_coord = float(get_param(gnbg_tmp_data, 'MinCoordinate'))
            problem_max_coord = float(get_param(gnbg_tmp_data, 'MaxCoordinate'))
            
            problem_comp_min_pos = get_param(gnbg_tmp_data, 'Component_MinimumPosition', is_scalar=False)
            problem_comp_sigma_flat = get_param(gnbg_tmp_data, 'ComponentSigma', is_scalar=False, dtype=np.float64)
            problem_comp_sigma = problem_comp_sigma_flat.flatten() if problem_comp_sigma_flat.ndim > 0 else np.array([problem_comp_sigma_flat.item()])


            problem_comp_h = get_param(gnbg_tmp_data, 'Component_H', is_scalar=False)
            problem_mu = get_param(gnbg_tmp_data, 'Mu', is_scalar=False)
            problem_omega = get_param(gnbg_tmp_data, 'Omega', is_scalar=False)
            
            problem_lambda_flat = get_param(gnbg_tmp_data, 'lambda', is_scalar=False)
            problem_lambda = problem_lambda_flat.flatten() if problem_lambda_flat.ndim > 0 else np.array([problem_lambda_flat.item()])


            problem_rot_matrix = get_param(gnbg_tmp_data, 'RotationMatrix', is_scalar=False)
            problem_opt_value = float(get_param(gnbg_tmp_data, 'OptimumValue'))
            problem_opt_pos = get_param(gnbg_tmp_data, 'OptimumPosition', is_scalar=False)

        except Exception as e:
            error_str = f"Error loading GNBG params for f{problem_idx} from {gnbg_data_path}: {e}\n{traceback.format_exc()}"
            print(error_str)
            continue

        problem_bounds_list = [(problem_min_coord, problem_max_coord)] * problem_dimension
        problem_lower_bounds = [b[0] for b in problem_bounds_list]
        problem_upper_bounds = [b[1] for b in problem_bounds_list]

        for run_rep in range(runs_per_gnbg_func_meta_eval):
            gnbg_instance = GNBG(
                MaxEvals=problem_max_fevals, AcceptanceThreshold=problem_accept_thresh,
                Dimension=problem_dimension, CompNum=problem_comp_num,
                MinCoordinate=problem_min_coord, MaxCoordinate=problem_max_coord,
                CompMinPos=problem_comp_min_pos, CompSigma=problem_comp_sigma,
                CompH=problem_comp_h, Mu=problem_mu, Omega=problem_omega,
                Lambda=problem_lambda, RotationMatrix=problem_rot_matrix,
                OptimumValue=problem_opt_value, OptimumPosition=problem_opt_pos
            )
            current_run_aocc = 0.0
            try:
                algorithm_instance = AlgorithmClass(
                    budget=problem_max_fevals, dim=problem_dimension,
                    lower_bounds=problem_lower_bounds, upper_bounds=problem_upper_bounds
                )
                # The optimize method should call gnbg_instance.fitness internally
                # and return best_solution_vector, best_fitness_value
                _best_sol_vec, _best_fitness_val = algorithm_instance.optimize(gnbg_instance.fitness)
                
                # Calculate AOCC for this run using the history from gnbg_instance
                current_run_aocc = calculate_aocc_from_gnbg_history(
                    gnbg_instance.FEhistory, 
                    gnbg_instance.OptimumValue, 
                    gnbg_instance.MaxEvals # Use the GNBG instance's MaxEvals as budget B for AOCC
                )
            except Exception as e:
                error_str = f"Error running algorithm '{algorithm_name}' on f{problem_idx}, run {run_rep+1}: {e}\n{traceback.format_exc()}"
                print(error_str)
                current_run_aocc = 0.0 # Penalize for error (worst AOCC)
            
            all_run_aocc_scores.append(current_run_aocc)

            if 1 <= problem_idx <= 6: unimodal_aoccs.append(current_run_aocc)
            elif 7 <= problem_idx <= 15: multimodal_single_aoccs.append(current_run_aocc)
            elif 16 <= problem_idx <= 24: multimodal_multi_aoccs.append(current_run_aocc)

    if not all_run_aocc_scores:
        overall_mean_aocc = 0.0
        overall_std_aocc = 0.0
        feedback_str = f"Algorithm {algorithm_name} could not be evaluated on any GNBG function."
        error_message = "No GNBG functions were successfully processed or all runs failed."
    else:
        overall_mean_aocc = np.mean(all_run_aocc_scores)
        overall_std_aocc = np.std(all_run_aocc_scores)
        feedback_str = (f"Algorithm {algorithm_name} achieved an average AOCC of {overall_mean_aocc:.4f} "
                        f"(higher is better, 1.0 is optimal) with std dev {overall_std_aocc:.4f} across GNBG functions.")
        error_message = ""

    if details:
        mean_unimodal = np.mean(unimodal_aoccs) if unimodal_aoccs else 0.0
        mean_multi_single = np.mean(multimodal_single_aoccs) if multimodal_single_aoccs else 0.0
        mean_multi_multi = np.mean(multimodal_multi_aoccs) if multimodal_multi_aoccs else 0.0
        feedback_str += (f"\n  Avg AOCC on Unimodal (f1-f6): {mean_unimodal:.4f}."
                         f"\n  Avg AOCC on Multimodal Single-Component (f7-f15): {mean_multi_single:.4f}."
                         f"\n  Avg AOCC on Multimodal Multi-Component (f16-f24): {mean_multi_multi:.4f}.")

    print(f"Evaluation for {algorithm_name}: Mean AOCC = {overall_mean_aocc:.4f}, Std Dev AOCC = {overall_std_aocc:.4f}")
    solution.add_metadata("gnbg_all_aocc_scores", all_run_aocc_scores)
    solution.set_scores(overall_mean_aocc, feedback_str, error_message) # LLaMEA will maximize this score

    return solution

# --- Task Prompt for GNBG ---
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
import numpy as np # Make sure to import numpy if you use it

class YourAlgorithmName:
    def __init__(self, budget, dim, lower_bounds, upper_bounds):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)
        
        self.eval_count = 0
        self.best_solution_so_far = None
        self.best_fitness_so_far = float('inf')
        # TODO: Add your algorithm's specific initializations here

    def _internal_evaluate(self, solution_batch_X, objective_function_ref):
        # Helper to manage evaluation calls and budget
        if self.eval_count >= self.budget:
            return np.full(solution_batch_X.shape[0], float('inf'))

        num_to_eval = min(solution_batch_X.shape[0], self.budget - self.eval_count)
        fitness_batch = np.full(solution_batch_X.shape[0], float('inf'))

        if num_to_eval > 0:
            actual_batch = solution_batch_X[:num_to_eval, :]
            try:
                evaluated_fitnesses = objective_function_ref(actual_batch)
                fitness_batch[:num_to_eval] = evaluated_fitnesses
            except Exception: # Catch errors from objective function if any
                pass 
            self.eval_count += num_to_eval
            for i in range(num_to_eval):
                if evaluated_fitnesses[i] < self.best_fitness_so_far:
                    self.best_fitness_so_far = evaluated_fitnesses[i]
                    self.best_solution_so_far = actual_batch[i, :].copy()
        return fitness_batch

    def optimize(self, objective_function):
        self.objective_function_ref = objective_function
        self.eval_count = 0
        self.best_solution_so_far = None
        self.best_fitness_so_far = float('inf')

        # --- YOUR NOVEL ALGORITHM'S CORE LOGIC GOES HERE ---
        # This is just a placeholder. The LLM should generate the actual algorithm.
        # For example, initialize a population:
        # pop_size = 50 
        # population = np.random.uniform(self.lower_bounds, self.upper_bounds, (pop_size, self.dim))
        # fitnesses = self._internal_evaluate(population, self.objective_function_ref)
        #
        # while self.eval_count < self.budget:
        #     # Implement selection, crossover, mutation, replacement
        #     # Ensure new solutions are evaluated using self._internal_evaluate
        #     # Ensure solutions are clipped to bounds
        #     if self.budget - self.eval_count == 0: break # Check budget
        #     # Example: generate one new random point if budget allows
        #     if self.eval_count < self.budget:
        #         new_point = np.random.uniform(self.lower_bounds, self.upper_bounds, (1,self.dim))
        #         self._internal_evaluate(new_point, self.objective_function_ref)
        #     else:
        #         break
        # --- END OF YOUR ALGORITHM'S CORE LOGIC ---
        
        # Fallback: if no evaluations happened or no best solution was set,
        # generate one random solution to return.
        if self.best_solution_so_far is None:
            if self.dim > 0 and self.lower_bounds.size > 0 and self.upper_bounds.size > 0:
                self.best_solution_so_far = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
                # This fallback solution is NOT evaluated here to save budget.
                # Its fitness will remain 'inf' unless an actual evaluation occurred.
            else: # Should not happen with proper initialization
                self.best_solution_so_far = np.array([0.0] * self.dim if self.dim > 0 else [0.0])
                self.best_fitness_so_far = float('inf')


        return self.best_solution_so_far, self.best_fitness_so_far
```
# Example of how the class might be used (for the LLM's understanding):
# bounds_example = [(-100.0, 100.0)] * 30
# optimizer = YourAlgorithmName(budget=500000, dim=30, lower_bounds=[b[0] for b in bounds_example], upper_bounds=[b[1] for b in bounds_example])
# def dummy_objective_function(X_batch): # X_batch is (N,D)
#     return np.sum(X_batch**2, axis=1) # Returns (N,)
# best_s, best_f = optimizer.optimize(dummy_objective_function)
"""

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- LLaMEA Configuration ---
    # IMPORTANT: Replace Gemini_LLM_Placeholder and LLaMEA_Placeholder with your actual class implementations
    
    # Use the placeholder LLM for now, replace with your actual Gemini_LLM
    llm_instance = Ollama_LLM(model="llama3.2:3b-instruct-fp16")    
    experiment_name_main = "llamea-gnbg-aocc-test-01"
    
    # Define where your GNBG .mat files (f1.mat to f24.mat) are located
    # Assumes they are in the same directory as this script for simplicity
    gnbg_mat_files_directory = os.path.dirname(os.path.abspath(__file__))

    print(f"LLaMEA Experiment: {experiment_name_main}")
    print(f"GNBG .mat files expected in: {gnbg_mat_files_directory}")

    # Create the LLaMEA instance
    # Note: LLaMEA's 'budget' is the number of algorithms it will generate and evaluate.
    # The 'budget' inside the LLM-generated algorithm's __init__ is from GNBG's MaxEvals.
    llamea_system = LLaMEA(
        f=lambda sol_obj, logger_obj: evaluateGNBG(
            sol_obj,
            explogger=logger_obj,
            details=True, # Get detailed feedback per GNBG category
            gnbg_problem_folder=gnbg_mat_files_directory,
            runs_per_gnbg_func_meta_eval=2 # Number of runs for EACH GNBG func during LLaMEA's eval
                                           # Keep this low (e.g., 1-3) for faster LLaMEA iterations.
                                           # For final competition, you'd run the *best resulting algorithm* 31 times.
        ),
        n_parents=1,  # For a (1+1)-LLaMEA strategy as in the paper
        n_offspring=1,
        llm=llm_instance,
        task_prompt=task_prompt_gnbg,
        experiment_name=experiment_name_main,
        elitism=True,    # True for (1+1) strategy
        HPO=False,       # Not using Hyperparameter Optimization in this setup
        budget=10,       # LLaMEA will generate and evaluate 10 algorithms in total
        minimization=False # LLaMEA will MAXIMIZE the score from evaluateGNBG (which is mean AOCC)
    )

    print("\n--- Starting LLaMEA run for GNBG using AOCC ---")
    best_overall_solution_object = llamea_system.run()
    print("\n--- LLaMEA run for GNBG Finished ---")

    if best_overall_solution_object and best_overall_solution_object.code != "# Placeholder":
        print(f"\nBest LLM-Generated Algorithm Found by LLaMEA:")
        print(f"Name: {best_overall_solution_object.name}")
        print(f"Description: {best_overall_solution_object.description}")
        print(f"Achieved Mean AOCC (Fitness): {best_overall_solution_object.fitness:.4f}")
        print(f"Feedback: {best_overall_solution_object.feedback}")
        if best_overall_solution_object.error_str:
            print(f"Errors during its evaluation: {best_overall_solution_object.error_str}")
        
        # To save the best algorithm's code:
        output_dir_for_best_algo = "best_llamea_gnbg_algo"
        os.makedirs(output_dir_for_best_algo, exist_ok=True)
        best_algo_filename = os.path.join(output_dir_for_best_algo, f"{best_overall_solution_object.name.replace(' ', '_')}.py")
        with open(best_algo_filename, "w") as f:
            f.write(f"# Algorithm: {best_overall_solution_object.name}\n")
            f.write(f"# Description: {best_overall_solution_object.description}\n")
            f.write(f"# Achieved Mean AOCC in LLaMEA: {best_overall_solution_object.fitness:.4f}\n")
            f.write(best_overall_solution_object.code)
        print(f"Code of the best algorithm saved to: {best_algo_filename}")
    else:
        print("LLaMEA run did not yield a valid best algorithm.")

