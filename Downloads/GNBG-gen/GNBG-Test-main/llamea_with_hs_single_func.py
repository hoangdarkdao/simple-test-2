import copy
import os

import concurrent.futures
import logging
import multiprocessing as mp
import random
import re
import traceback
import os, contextlib
import numpy as np
from ConfigSpace import ConfigurationSpace
from joblib import Parallel, delayed
from datetime import datetime
from llamea.solution import Solution
from llamea.loggers import ExperimentLogger
from llamea.utils import file_to_string
from my_utils.utils import extract_to_hs, extract_class_name_and_code
from harmony_search import HarmonySearchOptimizer
import dill
import joblib
joblib.parallel.DEFAULT_SERIALIZER = dill
folder = "logs/log_run_algorithms"
log_filename = f"{folder}/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def safe_evaluate_fitness(individual, f, logger, worst_value, timeout_seconds=60):
    def run_evaluation():
        try:
            return f(individual, logger)
        except Exception as e:
            return e

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_evaluation)
        try:
            result = future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            msg = f"[TIMEOUT] Evaluation exceeded {timeout_seconds} seconds and was skipped."
            individual.set_scores(worst_value, msg, msg)
       
            if hasattr(f, "log_individual"):
                f.log_individual(individual)
            return individual

    if isinstance(result, Exception):
        msg = f"[ERROR] Evaluation failed: {repr(result)}"
        individual.set_scores(worst_value, msg, msg)
        if hasattr(f, "log_individual"):
            f.log_individual(individual)
        return individual

    return result


class LLaMEA: # with key. rotations

    def __init__(self,
        f, llms: list,
        init_pop_size=2, pop_size=2, mutation_rate = 0.3,
        experiment_name="",
        budget=100,
        eval_timeout=3600,
        log=True,
        minimization=False):
        self.default_llm = llms[0]
        self.crossover_llm = llms[1]
        self.mutate_llm = llms[2]
        self.llms = llms
        self.eval_timeout = eval_timeout
        self.f = f  # evaluation function, provides an individual as output.        
        self.mutation_rate = mutation_rate
        self.budget = budget
        self.init_pop_size = init_pop_size
        self.pop_size = pop_size
        self.population = []
        self.generation = 0
        self.run_history = []
        self.log = log
        self.minimization = minimization
        self.worst_value = -np.Inf
        if minimization:
            self.worst_value = np.Inf
        self.best_so_far = Solution(name="", code="")
        self.best_so_far.set_scores(self.worst_value, "", "")
        self.experiment_name = experiment_name
        self.elitist = None # the best solution object found
        if self.log:
            # modelname = self.model.replace(":", "_")
            self.logger = ExperimentLogger(f"LLaMEA--{experiment_name}")
            self.default_llm.set_logger(self.logger)
            self.crossover_llm.set_logger(self.logger)
            self.mutate_llm.set_logger(self.logger)
        else:
            self.logger = None
        self.textlog = logging.getLogger(__name__)
       
        self.reflection_prompt = file_to_string("prompt/reflective_prompt_single_func.txt")
        self.crossover_prompt = file_to_string("prompt/crossover.txt")
        self.role_prompt = file_to_string("prompt/role_generator_func_3.txt")
        self.task_prompt = file_to_string("prompt/task_output_generator.txt")
        self.mutation_prompt = file_to_string("prompt/mutation.txt")
        self.comprehensive_reflection_prompt = file_to_string("prompt/comprehensive_reflection.txt")
        
        self.str_comprehensive_memory = ""
        self.good_reflections_list = []
        self.bad_reflections_list = []
        self.harmony_optimizer = HarmonySearchOptimizer(f, self.default_llm)

    def logevent(self, event):
        self.textlog.info(event)

    def mutate(self) -> list[Solution]:
        full_mutation_prompt = self.mutation_prompt.format(
            user_generator = self.role_prompt, 
            func_signature1 = self.best_so_far.name,
            elitist_code = self.best_so_far.code,
            reflection = self.str_comprehensive_memory
        )
        session_messages = [
            {
                "role": "user", 
                "content": full_mutation_prompt + self.task_prompt
            },
        ]
        logging.info("Mutation prompt: " + full_mutation_prompt)
        offsprings = [self.mutate_llm.sample_solution(copy.deepcopy(session_messages)) for _ in range(int(self.pop_size * self.mutation_rate))] # generate 5 mutated version of the best solution
        print("Sample in mutation sucessfully!")
        evaluated_offsprings = []
        for offspring in offsprings:
            offspring.generation = self.generation
            # offspring = self.evaluate_fitness(offspring)
            evaluated_offsprings.append(offspring)
        evaluated_offsprings = self.evaluate_population_parallel(evaluated_offsprings)
        return evaluated_offsprings
    
    def initialize_population_from_seeds(self):
        # seed_dir = "seed_algorithms_func_3"
        # self.seed_files = sorted(
        #     [os.path.join(seed_dir, f) for f in os.listdir(seed_dir) if f.endswith(".py")]
        # )
        self.seed_files = ['seed_algorithms/group3_func.py']
        self.textlog.info(f"Initializing population from {len(self.seed_files)} seed files...")
        initial_solutions = []
        for file_path in self.seed_files:
            try:
                with open(file_path, 'r') as f:
                    code = f.read()
                # A simple way to get the class name from the file
                match = re.search(r'class\s+(\w+)', code)
                name = match.group(1)
                sol = Solution(code=code, name=name, generation=self.generation, description=f"Seed from {name}")
                self.population.append(sol)
                self.run_history.append(sol)
            except Exception as e:
                self.textlog.error(f"Failed to load seed file: {file_path}", exc_info=True)
        self.population = self.evaluate_population_parallel(self.population)
        
        self.update_best()

    def initialize_single(self, role_index: int):
        """
        Initializes a single solution.
        """
      

        current_role_prompt = self.role_prompt[role_index % len(self.role_prompt)]
        new_individual = Solution(name="", code="", generation=self.generation)
        session_messages = [
            {
                "role": "user",
                "content": self.role_prompt
                + self.task_prompt
            },
        ]
       
        try:
            new_individual = self.default_llm.sample_solution(session_messages, role_index=role_index)
            new_individual.generation = self.generation
            # new_individual = self.evaluate_fitness(new_individual)
        except Exception as e:
            new_individual.set_scores(
                self.worst_value,
                f"An exception occured: {traceback.format_exc()}.",
                repr(e) + traceback.format_exc(),
            )
            self.logevent(f"An exception occured: {traceback.format_exc()}.")
            if hasattr(self.f, "log_individual"):
                self.f.log_individual(new_individual)
        
        return new_individual # = Solution class
    
    def evaluate_population_parallel(self, population: list[Solution], n_jobs=2):
        evaluated_population = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(safe_evaluate_fitness)(
                ind, self.f, self.logger, self.worst_value
            ) for ind in population
        )
        return evaluated_population
    
    def initialize(self):
        """
        Initializes the evolutionary process by generating the first parent population.
        """
        self.initialize_population_from_seeds()
        population = []
        population_gen = []
        try:
            timeout = self.eval_timeout
            population_gen = Parallel( # maybe this
                n_jobs=2,
                backend="loky",
                timeout=timeout + 15,
                return_as="generator_unordered",
            )(delayed(self.initialize_single)(i) for i in range(self.init_pop_size - len(self.population)))
        except Exception as e:
            print(f"Parallel time out in initialization {e}, retrying.")

        for p in population_gen:
            self.run_history.append(p)  # update the history
            population.append(p)
        population = self.evaluate_population_parallel(population)
        self.generation += 1
        self.population.extend(population)  # Save the entire population
        self.update_best()

    def flash_reflection(self, selected_population: list[Solution]):
        chosen_llm = self.mutate_llm 
        self.textlog.info(f"--- Performing Long-Term Reflection at Generation {self.generation} ---")
        sorted_population = sorted(
            [p for p in self.population if np.isfinite(p.fitness)], 
            key=lambda x: x.fitness, 
            reverse=not self.minimization
        )
        if not sorted_population:
            self.textlog.warning("Skipping reflection: No valid individuals in population.")
            return
        
        lst_method_str = ""
        for i, sol in enumerate(sorted_population):
            lst_method_str += f"### Rank {i+1} (Overall AOCC Score on Multimodal instances with multiple components: {sol.fitness:.4e}"
            lst_method_str += f"# Name: {sol.name}\n"
            lst_method_str += f"# Description: {sol.description}\n"
            lst_method_str += f"# Code:\n```python\n{sol.code}\n```\n\n"
            
        full_reflection_prompt = self.reflection_prompt.format(problem_desc = self.role_prompt,
                                                               lst_method=lst_method_str)
        logging.info(f"Reflection Prompt: {full_reflection_prompt}")
        try:
            # It's recommended to use a powerful model for this reasoning task if possible
            response_text = chosen_llm.query([{"role": "user", "content": full_reflection_prompt}])
            self.textlog.info(f"Full response text: {response_text}")
            # 4. Parse the response and update the strategy
            if response_text and "**Experience:**" in response_text and "**Analysis:**" in response_text:
                # Extract the text after "**Experience:**"
                analyze_start = response_text.find("**Analysis:**") + len("**Analysis:**")
                exp_start = response_text.find("**Experience:**")
                
                analysis_text = response_text[analyze_start:exp_start].strip()
                experience_text = response_text[exp_start + len("**Experience:**"):].strip()
                
                flash_reflection_json = {
                    "analyze": analysis_text,
                    "exp": experience_text
                }
                
                self.str_flash_memory = flash_reflection_json # user this later in comprehensive reflection, to compare this with previous reflection
                file_name = f"reflection/problem_iter_flash_reflection.txt"
                with open(file_name, 'w') as file:
                    file.writelines(response_text)
            else:
                self.textlog.warning("Long-term reflection response did not contain 'Experience:'. Using previous strategy.")
        except Exception as e:
            self.textlog.error(f"Failed to perform long-term reflection: {e}", exc_info=True)
    
    def comprehensive_reflection(self):
        good_reflection = '\n\n'.join(self.good_reflections_list) if len(self.good_reflections_list) > 0 else "None"
        bad_reflection = '\n\n'.join(self.bad_reflections_list) if len(self.bad_reflections_list) > 0 else "None"

        full_comprehensive_reflection_prompt = self.comprehensive_reflection_prompt.format(
            bad_reflection = bad_reflection,
            good_reflection = good_reflection,
            curr_reflection = self.str_flash_memory['exp'],
        )
        
        response_text = self.mutate_llm.query([{"role": "user", "content": full_comprehensive_reflection_prompt}])
        self.textlog.info(f"Full response text: {response_text}")
        self.str_comprehensive_memory = response_text

    def crossover(self, parents: list[Solution]) -> list[Solution]:
        if not parents:
            self.textlog.warning("Crossover skipped: No parents provided.")
            return []
        if len(parents) % 2 != 0:
            parents.pop()
        crossed_population = []
        self.textlog.info(f"Generating offspring via Crossover...")
        offsprings = []
        for i in range(0, len(parents), 2):
            # Select two individuals
            if parents[i].fitness < parents[i + 1].fitness:
                parent_1 = parents[i]
                parent_2 = parents[i + 1]
            else:
                parent_1 = parents[i + 1]
                parent_2 = parents[i]
            
            full_crossover_prompt = self.crossover_prompt.format(
                role = self.role_prompt,
                func_signature_m1 = parent_1.name,
                func_signature_m2 = parent_2.name,
                code_method1 = parent_1.code,
                code_method2 = parent_2.code,
                analyze = self.str_flash_memory['analyze'], # str flash is a json with 2 keys
                exp = self.str_comprehensive_memory
            )
            # print(f"Full: {full_crossover_prompt}")
            session_messages = [
                {
                    "role": "user", 
                    "content": full_crossover_prompt + self.task_prompt
                },
            ]
            offspring = self.crossover_llm.sample_solution(session_messages)
            print("Sample in crossover sucessfully!")
            offspring.generation = self.generation
            offsprings.append(offspring)
            # offspring = self.evaluate_fitness(offspring)
            # crossed_population.append(offspring)
        offsprings = self.evaluate_population_parallel(offsprings)
        crossed_population.extend(offsprings)
        
        assert len(crossed_population) == self.pop_size, "Crossed population does not equal to population size"
        logging.info("Crossover Prompt: " + full_crossover_prompt)
        return crossed_population # = pop_size         

    def update_best(self):
        """
        Update the best individual in the new population
        """
        if self.minimization == False:
            best_individual = max(self.population, key=lambda x: x.fitness)

            if best_individual.fitness > self.best_so_far.fitness:
                self.best_so_far = best_individual
        else:
            best_individual = min(self.population, key=lambda x: x.fitness)

            if best_individual.fitness < self.best_so_far.fitness:
                self.best_so_far = best_individual
        print(f"Best individual fitness is: {best_individual.fitness}")
        
    def random_select(self, population: list[Solution]) -> list[Solution]:
        
        selected_population = []
        if len(population) < 2:
            return None
        trials = 0
        while len(selected_population) < 2 * self.pop_size:
            trials += 1
            parents = np.random.choice(population, size=2, replace=False)
            if parents[0].fitness != parents[1].fitness:
                selected_population.extend(parents)
            if trials > 1000:
                print("Exceed valid trials")
                return None
        return selected_population # selected_population = 2 * pop_size

    def tournament_selection(self, population: list[Solution], k=3, num_winners=None) -> list[Solution]:
        """
        Perform tournament selection to choose a subset of individuals.
        """
        if not population:
            self.textlog.warning("Tournament selection skipped: Empty population.")
            return []

        valid_pop = [s for s in population if hasattr(s, 'fitness') and np.isfinite(s.fitness)]
        if not valid_pop:
            return []

        if num_winners is None:
            num_winners = self.pop_size * 2

        selected = []
        for _ in range(num_winners):
            contenders = random.sample(valid_pop, min(k, len(valid_pop)))
            if self.minimization:
                winner = min(contenders, key=lambda x: x.fitness)
            else:
                winner = max(contenders, key=lambda x: x.fitness)
            selected.append(winner)

        return selected

    def run(self): 
       
        self.logevent("Initializing first population")
        self.initialize()  # Initialize a population

        if self.log:
            self.logger.log_population(self.population)

        self.logevent(
            f"Started evolutionary loop, best so far: {self.best_so_far.fitness}"
        )
        while len(self.run_history) < self.budget:

            population_to_select = self.population if (self.best_so_far is None or self.best_so_far in self.population) else [
                                                                                                                         self.best_so_far] + self.population  # add elitist to population for selection
            selected_population = self.tournament_selection(population_to_select)
            logging.info(f"Population length is: {len(population_to_select)}")
            if not selected_population:
                print("Skipping this generation due to no valid selection.")
                continue
            self.flash_reflection(selected_population) 
            self.comprehensive_reflection()
            
            curr_good_code = self.best_so_far # best solution before update
            
            crossed_population = self.crossover(selected_population) # len(crossed_population) = len(self.population)
            self.logger.log_population(crossed_population)
            self.population = crossed_population
            self.update_best()
            mutated_population = self.mutate() # get a list of mutation solution of best so far
            self.logger.log_population(mutated_population)
            self.population.extend(mutated_population)
            self.update_best()
            
            self.population = sorted(self.population, key=lambda x: x.fitness)[:self.pop_size] # without this the population size will increase after each iteration

            self.run_history.extend(self.population)
              
            if self.log:
                self.logger.log_population(self.population)
            
            if curr_good_code.name != self.best_so_far.name:
                self.good_reflections_list.append(self.str_flash_memory['exp'])
                with open(f"reflection/good_reflection/good_gen{self.generation}.txt", "w") as f:
                    f.write(self.str_flash_memory['exp'])
            else:
                self.bad_reflections_list.append(self.str_flash_memory['exp'])
                with open(f"reflection/bad_reflection/bad_gen{self.generation}.txt", "w") as f:
                    f.write(self.str_flash_memory['exp'])
                    
            with open(f"reflection/comprehensive_reflection/comp_gen{self.generation}.txt", "w") as f:
                f.write(self.str_comprehensive_memory)
            logging.info("Perform Harmony Search...")
            try_hs_num = 3
            while try_hs_num:
                individual_hs = self.harmony_optimizer.run_hs(self.best_so_far)
                
                if individual_hs is not None:
                    self.logger.log_population([individual_hs])
                    self.population.extend([individual_hs])
                        # self.update_iter()
                    break
                else:
                    try_hs_num -= 1
            self.update_best()
            self.logevent(
                f"Generation {self.generation}, best so far: {self.best_so_far.fitness}"
            )
            self.generation += 1
        return self.best_so_far
