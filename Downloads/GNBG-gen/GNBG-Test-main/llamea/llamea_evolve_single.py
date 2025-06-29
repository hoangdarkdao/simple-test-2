"""LLaMEA - LLM powered Evolutionary Algorithm for code optimization
This module integrates OpenAI's language models to generate and evolve
algorithms to automatically evaluate (for example metaheuristics evaluated on BBOB).
"""
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
from llamea.utils import NoCodeException, handle_timeout, discrete_power_law_distribution, file_to_string
from prompt.multi_role_prompts import *

# TODOs:
# Implement diversity selection mechanisms (none, prefer short code, update population only when (distribution of) results is different, AST / code difference)
folder = "log/log_run_algorithms"
log_filename = f"{folder}/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


class LLaMEA: # with key. rotations
    """
    A class that represents the Language Model powered Evolutionary Algorithm (LLaMEA).
    This class handles the initialization, evolution, and interaction with a language model
    to generate and refine algorithms.
    """

    def __init__(
        self,
        f,
        llms: list,
        n_parents=5,
        n_offspring=10,
        experiment_name="",
        elitism=False,
        HPO=False,
        mutation_prompts=None,
        adaptive_mutation=False,
        budget=100,
        eval_timeout=3600,
        max_workers=10,
        log=True,
        minimization=False,
        _random=False,
    ):
       
        self.llms = llms
        self.llm_index = 0
        self.eval_timeout = eval_timeout
        self.f = f  # evaluation function, provides an individual as output.
     
       
        self.mutation_prompts = mutation_prompts
        self.adaptive_mutation = adaptive_mutation
        if mutation_prompts == None:
            self.mutation_prompts = [
                "Refine the strategy of the selected solution to improve it.",  # small mutation
                # "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
            ]
        self.budget = budget
        self.n_parents = n_parents
        self.n_offspring = n_offspring
        self.population = []
        self.elitism = elitism
        self.generation = 0
        self.run_history = []
        self.log = log
        self._random = _random
        self.HPO = HPO
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
            self.llms[self.llm_index].set_logger(self.logger)
        else:
            self.logger = None
        self.textlog = logging.getLogger(__name__)
        if max_workers > self.n_offspring:
            max_workers = self.n_offspring
        self.max_workers = max_workers
        # Define prompt
        self.reflection_prompt = file_to_string("prompt/reflective_prompt.txt")
        self.crossover_prompt = file_to_string("prompt/crossover.txt")
        self.role_prompt = file_to_string("prompt/role_generator.txt")
        self.task_prompt = file_to_string("prompt/task_output_generator.txt")
        self.comprehensive_reflection_prompt = file_to_string("prompt/comprehensive_reflection.txt")
        self.str_comprehensive_memory = ""
        
    def _get_next_llm(self):
        """Cycles through the list of LLM instances in a round-robin fashion."""
        llm_instance = self.llms[self.llm_index]
        self.textlog.info(f"Using LLM instance #{self.llm_index} (Model: {llm_instance.model})")
        self.llm_index = (self.llm_index + 1) % len(self.llms)
        return llm_instance

    def logevent(self, event):
        self.textlog.info(event)


    def initialize_single(self, role_index: int):
        """
        Initializes a single solution.
        """
        chosen_llm = self.llms[role_index % len(self.llms)] 
        self.textlog.info(f"Using LLM api key #{chosen_llm.api_key})")

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
            new_individual = chosen_llm.sample_solution(session_messages, role_index=role_index)
            new_individual.generation = self.generation
            new_individual = self.evaluate_fitness(new_individual)
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

    def initialize(self):
        """
        Initializes the evolutionary process by generating the first parent population.
        """
        population = []
        population_gen = []
        try:
            timeout = self.eval_timeout
            population_gen = Parallel( # maybe this
                n_jobs=2,
                backend="loky",
                timeout=timeout + 15,
                return_as="generator_unordered",
            )(delayed(self.initialize_single)(i) for i in range(self.n_parents))
        except Exception as e:
            print(f"Parallel time out in initialization {e}, retrying.")

        for p in population_gen:
            self.run_history.append(p)  # update the history
            population.append(p)

        self.generation += 1
        self.population.extend(population)  # Save the entire population
        self.update_best()

    def flash_reflection(self, selected_population: list[Solution]):
        chosen_llm = self.llms[0] 
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
            lst_method_str += f"### Rank {i+1} (AOCC Score: {sol.fitness:.4e})\n"
            lst_method_str += f"# Name: {sol.name}\n"
            lst_method_str += f"# Description: {sol.description}\n"
            lst_method_str += f"# Code:\n```python\n{sol.code}\n```\n\n"
            
        full_reflection_prompt = self.reflection_prompt.format(problem_desc = self.role_prompt,
                                                               lst_method=lst_method_str)
        print(f"Reflection Prompt: {full_reflection_prompt}")
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
    
    # def comprehensive_reflection(self):
    #     user_prompt = self.comprehensive_reflection_prompt.format(
    #         curr_reflection = self.str_flash_memory['exp'],
    #     )
    def evaluate_fitness(self, individual):
        """
        Evaluates the fitness of the provided individual by invoking the evaluation function `f`.
        This method handles error reporting and logs the feedback, fitness, and errors encountered.

        Args:
            individual (dict): Including required keys "_solution", "_name", "_description" and optional "_configspace" and others.

        Returns:
            tuple: Updated individual with "_feedback", "_fitness" (float), and "_error" (string) filled.
        """
        timeout_seconds = 60

        def run_evaluation():
            try:
                return self.f(individual, self.logger)
            except Exception as e:
                return e

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_evaluation)
            try:
                result = future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                msg = f"[TIMEOUT] Evaluation exceeded {timeout_seconds} seconds and was skipped."
                individual.set_scores(self.worst_value, msg, msg)
                self.logevent(msg)
                if hasattr(self.f, "log_individual"):
                    self.f.log_individual(individual)
                return individual

        if isinstance(result, Exception):
            msg = f"[ERROR] Evaluation failed: {repr(result)}"
            individual.set_scores(self.worst_value, msg, msg)
            self.logevent(msg)
            if hasattr(self.f, "log_individual"):
                self.f.log_individual(individual)
            return individual

        return result





    def crossover(self, parents: list[dict]) -> list[dict]:
        if len(parents) % 2 != 0:
            parents.pop()
        offspring_list = []
        chosen_llm = self.llms[0]
        self.textlog.info(f"Generating offspring via Crossover...")
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
                exp = self.str_flash_memory['exp']
            )
            # print(f"Full: {full_crossover_prompt}")
            session_messages = [
                {
                    "role": "user", 
                    "content": full_crossover_prompt + self.task_prompt
                },
            ]
            offspring = chosen_llm.sample_solution(session_messages)
            print("Sample in crossover sucessfully!")
            offspring.generation = self.generation
            offspring = self.evaluate_fitness(offspring)
            offspring_list.append(offspring)
        logging.info("Crossover Prompt: " + full_crossover_prompt)
        return offspring_list            

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
        
    def selection(self, parents, offspring): # sort the parents and offsprings and choose the new population = parent length
        """
        Select the new population based on the parents and the offspring and the current strategy.

        Args:
            parents (list): List of solutions.
            offspring (list): List of new solutions.

        Returns:
            list: List of new selected population.
        """
        print(f"parent: {parents}")
        reverse = self.minimization == False

        # TODO filter out non-diverse solutions
        if self.elitism:
            # Combine parents and offspring
            combined_population = parents + offspring
            # Sort by fitness
            combined_population.sort(key=lambda x: x.fitness, reverse=reverse)
            # Select the top individuals to form the new population
            new_population = combined_population[: 5]
        else:
            # Sort offspring by fitness
            offspring.sort(key=lambda x: x.fitness, reverse=reverse)
            # Select the top individuals from offspring to form the new population
            new_population = offspring[: self.n_parents]

        print(f"After selection, new population is: {new_population}")
        return new_population

    def evolve_solution(self, individual, worker_id: int):
        """
        Evolves a single solution by constructing a new prompt,
        querying the LLM, and evaluating the fitness.
        """
        chosen_llm = self.llms[worker_id % len(self.llms)]
        self.textlog.info(f"Using LLM api key #{chosen_llm.api_key})")

        new_prompt = self.construct_prompt(individual)
        evolved_individual = individual.copy()

        try:
            evolved_individual = chosen_llm.sample_solution(
                new_prompt, evolved_individual.parent_ids, HPO=self.HPO
            )
            evolved_individual.generation = self.generation
            evolved_individual = self.evaluate_fitness(evolved_individual)
        except Exception as e:
            error = repr(e)
            evolved_individual.set_scores(
                self.worst_value, f"An exception occurred: {error}.", error
            )
            if hasattr(self.f, "log_individual"):
                self.f.log_individual(evolved_individual)
            self.logevent(f"An exception occured: {traceback.format_exc()}.")

        # self.progress_bar.update(1)
        return evolved_individual

    def run(self): 
        """ 
        initialization -> selection -> flash reflection -> comprehensive reflection -> crossover -> elitist mutation
        flash reflection = 
        1. comprehensive description on ranking pairs
        2. compare the analysis at time step t to that of time step t-1 => guide information
        
        Returns:
            tuple: A tuple containing the best solution and its fitness at the end of the evolutionary process.
        """
        self.logevent("Initializing first population")
        self.initialize()  # Initialize a population

        if self.log:
            self.logger.log_population(self.population)

        self.logevent(
            f"Started evolutionary loop, best so far: {self.best_so_far.fitness}"
        )
        while len(self.run_history) < self.budget:

            population_to_select = self.population
            selected_population = self.selection(self.population, population_to_select) # if choose as the parents length
            self.flash_reflection(selected_population) 
            offsprings = self.crossover(selected_population)
            
            for p in offsprings:
                self.run_history.append(p)
                print(f"Run history: {self.run_history}") # run history [1, 2, 3]
                print(f"New_population: {offsprings}") # return 3, as 3 is the latest
            self.generation += 1
            if self.log:
                self.logger.log_population(offsprings)
            self.population = self.selection(self.population, offsprings)
            self.update_best()
            self.logevent(
                f"Generation {self.generation}, best so far: {self.best_so_far.fitness}"
            )
            
        return self.best_so_far
