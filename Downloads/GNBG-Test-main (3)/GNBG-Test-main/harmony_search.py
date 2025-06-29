import numpy as np
from llamea.solution import Solution
from my_utils.utils import extract_class_name_and_code, file_to_string, extract_to_hs
import logging
import os
from datetime import datetime
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)

log_file = os.path.join(log_folder, f"harmony_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    filename=log_file,
    filemode='w',  # 'w' to overwrite each time, or 'a' to append
    level=logging.INFO,  # or DEBUG, WARNING, etc.
    format='%(asctime)s - %(levelname)s - %(message)s'
) 

class HarmonySearchOptimizer:
    def __init__(self, f_evaluate, llm,
            hs_size = 5, hs_max_iter = 5, hmcr = 0.7, par = 0.5, bandwidth = 0.2, str_input = False):
        self.f_evaluate = f_evaluate; self.llm = llm # GNBG.fitness
        self.hs_size = hs_size
        self.hs_max_iter = hs_max_iter
        self.hmcr = hmcr
        self.str_input = str_input
        self.par = par
        self.bandwidth = bandwidth
        self.hs_prompt = file_to_string("prompt/harmony_search.txt")

    def initialize_harmony_memory(self, bounds: list): # parameters bound of best found solution
        problem_size = len(bounds)
        memory = np.zeros((self.hs_size, problem_size))
        for i in range(problem_size):
            memory[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], self.hs_size)
        return memory
    
    def create_population_hs(self, str_code: str, parameter_ranges: dict, harmony_memory) -> list[Solution]:
        hs_pop = []
        print(f"Parameter range: {parameter_ranges}")
        print(f"Harmony memory: {harmony_memory}")
        for i in range(len(harmony_memory)): # harmony_memory: (hs_size, problem_size)
            tmp_str = str_code
            for j in range(len(list(parameter_ranges))):
                tmp_str = tmp_str.replace(('{' + list(parameter_ranges)[j] + '}'), str(harmony_memory[i][j]))
                if tmp_str == str_code:
                    logging.warning("No replacements made in template string. Returning None.")
                    return None
            name, code = extract_class_name_and_code(tmp_str)
            temp_sol = Solution(name=name, code=code)
            temp_sol = self.f_evaluate(temp_sol)
            print(f"Evaluated individual {name} with fitness {temp_sol.fitness}")
            print(f"New algorithm: {temp_sol.code}")
            hs_pop.append(temp_sol)
        return hs_pop 

    def create_new_harmony(self, harmony_memory, bounds: list):
        new_harmony = np.zeros((harmony_memory.shape[1],))
        for i in range(harmony_memory.shape[1]):
            if np.random.rand() < self.hmcr:
                new_harmony[i] = harmony_memory[np.random.randint(0, harmony_memory.shape[0]), i]
                if np.random.rand() < self.par:
                    adjustment = np.random.uniform(-1, 1) * (bounds[i][1] - bounds[i][0]) * self.bandwidth
                    new_harmony[i] += adjustment
            else:
                new_harmony[i] = np.random.uniform(bounds[i][0], bounds[i][1])
                
        logging.debug(f"Created new harmony: {new_harmony}")
        return new_harmony
    
    def update_harmony_memory(self, population_hs, harmony_memory, new_harmony, func_block, parameter_ranges):
        objs = [individual.fitness for individual in population_hs]
        worst_index = np.argmin(np.array(objs)) # index i

        new_individual = self.create_population_hs(func_block, parameter_ranges, [new_harmony.tolist()])[0]

        if new_individual.fitness > population_hs[worst_index].fitness:
            print(f"HS Update: New harmony (fitness {new_individual.fitness:.4e}) is better than worst ({population_hs[worst_index].fitness:.4e}). Replacing.")

            population_hs[worst_index] = new_individual
            harmony_memory[worst_index] = new_harmony
        return population_hs, harmony_memory

    def run_hs(self, elitist: Solution, code_str = None):
                 # response_text = chosen_llm.query([{"role": "user", "content": full_reflection_prompt}])
        if self.str_input:
            full_hs_prompt = self.hs_prompt.format(code_extract = code_str) 
        else:
           full_hs_prompt = self.hs_prompt.format(code_extract = elitist.code) 
        responses = self.llm.query([{"role": "user", "content": full_hs_prompt}])
        parameter_ranges, func_block = extract_to_hs(responses)
        if parameter_ranges is None or func_block is None:
            return None
        bounds = [value for value in parameter_ranges.values()] # [(100, 10), (20, 30), ..]

        harmony_memory = self.initialize_harmony_memory(bounds) # (hs_size, problem_size)
        population_hs = self.create_population_hs(func_block, parameter_ranges, harmony_memory)
        if population_hs is None:
            return None
        # # elif len([individual for individual in population_hs if individual["exec_success"] is True]) == 0:
        # #     self.function_evals -= self.cfg.hm_size
        # #     return None
        for iteration in range(self.hs_max_iter):
            new_harmony = self.create_new_harmony(harmony_memory, bounds) # bounds: list of para range in a function
            print(f"New harmony is: {new_harmony}")
            population_hs, harmony_memory = self.update_harmony_memory(population_hs, harmony_memory, new_harmony,
                                                                       func_block, parameter_ranges)
            print(f"")
        best_individual = max(population_hs, key = lambda x : x.fitness)
        best_individual.try_hs = True
        return best_individual