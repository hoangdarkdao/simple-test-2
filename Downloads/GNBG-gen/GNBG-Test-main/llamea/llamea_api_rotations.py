"""LLaMEA - LLM powered Evolutionary Algorithm for code optimization
This module integrates OpenAI's language models to generate and evolve
algorithms to automatically evaluate (for example metaheuristics evaluated on BBOB).
"""
import concurrent.futures
import logging
import random
import re
import traceback
import os, contextlib
import numpy as np
from ConfigSpace import ConfigurationSpace
from joblib import Parallel, delayed
from datetime import datetime

from .solution import Solution
from .loggers import ExperimentLogger
from .utils import NoCodeException, handle_timeout, discrete_power_law_distribution
from prompt.multi_role_prompts import *

# TODOs:
# Implement diversity selection mechanisms (none, prefer short code, update population only when (distribution of) results is different, AST / code difference)

log_filename = f"log_run_algorithms/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
        role_prompt="",
        task_prompt="",
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
        """
        Initializes the LLaMEA instance with provided parameters. Note that by default LLaMEA maximizes the objective.

        Args:
            f (callable): The evaluation function to measure the fitness of algorithms.
            n_parents (int): The number of parents in the population.
            n_offspring (int): The number of offspring each iteration.
            elitism (bool): Flag to decide if elitism (plus strategy) should be used in the evolutionary process or comma strategy.
            role_prompt (str): A prompt that defines the role of the language model in the optimization task.
            task_prompt (str): A prompt describing the task for the language model to generate optimization algorithms.
            experiment_name (str): The name of the experiment for logging purposes.
            elitism (bool): Flag to decide if elitism should be used in the evolutionary process.
            HPO (bool): Flag to decide if hyper-parameter optimization is part of the evaluation function.
                In case it is, a configuration space should be asked from the LLM as additional output in json format.
            mutation_prompts (list): A list of prompts to specify mutation operators to the LLM model. Each mutation, a random choice from this list is made.
            adaptive_mutation (bool): If set to True, the mutation prompt 'Change X% of the lines of code' will be used in an adaptive control setting.
                This overwrites mutation_prompts.
            budget (int): The number of generations to run the evolutionary algorithm.
            eval_timeout (int): The number of seconds one evaluation can maximum take (to counter infinite loops etc.). Defaults to 1 hour.
            log (bool): Flag to switch of the logging of experiments.
            minimization (bool): Whether we minimize or maximize the objective function. Defaults to False.
            _random (bool): Flag to switch to random search (purely for debugging).
        """
        self.llms = llms
        self.llm_index = 0
        self.i = 0 # circle the role prompt around
        self.eval_timeout = eval_timeout
        self.f = f  # evaluation function, provides an individual as output.
        self.role_prompt = role_prompt
        if role_prompt == "":
            self.role_prompt = [role1]
        if task_prompt == "":
            self.task_prompt = "Hihi"
        else:
            self.task_prompt = task_prompt
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
                "content": current_role_prompt
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
        
        return new_individual

    def initialize(self):
        """
        Initializes the evolutionary process by generating the first parent population.
        """
       
        population = []
        population_gen = []
        try:
            timeout = self.eval_timeout
            population_gen = Parallel( # maybe this
                n_jobs=self.max_workers,
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
        self.population = population  # Save the entire population
        self.update_best()

    def evaluate_fitness(self, individual):
        """
        Evaluates the fitness of the provided individual by invoking the evaluation function `f`.
        This method handles error reporting and logs the feedback, fitness, and errors encountered.

        Args:
            individual (dict): Including required keys "_solution", "_name", "_description" and optional "_configspace" and others.

        Returns:
            tuple: Updated individual with "_feedback", "_fitness" (float), and "_error" (string) filled.
        """
        updated_individual = self.f(individual, self.logger)
        print(f"Update_individual is: {updated_individual}")

        return updated_individual

    def construct_prompt(self, individual):
        """
        Constructs a new session prompt for the language model based on a selected individual.

        Args:
            individual (dict): The individual to mutate.

        Returns:
            list: A list of dictionaries simulating a conversation with the language model for the next evolutionary step.
        """
        # Generate the current population summary
        current_role_prompt = self.role_prompt[individual.role_prompt_index % len(self.role_prompt)]
        population_summary = "\n".join([ind.get_summary() for ind in self.population])
        solution = individual.code
        description = individual.description
        feedback = individual.feedback
        if self.adaptive_mutation == True:
            num_lines = len(solution.split("\n"))
            prob = discrete_power_law_distribution(num_lines, 1.5)
            new_mutation_prompt = f"""Refine the strategy of the selected solution to improve it. 
Make sure you only change {(prob*100):.1f}% of the code, which means if the code has 100 lines, you can only change {prob*100} lines, and the rest of the lines should remain unchanged. 
This input code has {num_lines} lines, so you can only change {max(1, int(prob*num_lines))} lines, the rest {num_lines-max(1, int(prob*num_lines))} lines should remain unchanged. 
This changing rate {(prob*100):.1f}% is a mandatory requirement, you cannot change more or less than this rate.
"""
            self.mutation_prompts = [new_mutation_prompt]

        mutation_operator = random.choice(self.mutation_prompts)
        individual.set_operator(mutation_operator)

        final_prompt = f"""{self.task_prompt}
The current population of algorithms already evaluated (name, description, score) is:
{population_summary}

The selected solution to update is:
{description}

With code:
{solution}

{feedback}

{mutation_operator}
"""
        print(f"Feedback is: {feedback}")
        session_messages = [
            {"role": "user", "content": current_role_prompt + final_prompt},
        ]

        if self._random:  # not advised to use, only for debugging purposes
            session_messages = [
                {"role": "user", "content": self.task_prompt},
            ]
        # Logic to construct the new prompt based on current evolutionary state.
        return session_messages

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
    def selection(self, parents, offspring):
        """
        Select the new population based on the parents and the offspring and the current strategy.

        Args:
            parents (list): List of solutions.
            offspring (list): List of new solutions.

        Returns:
            list: List of new selected population.
        """
        reverse = self.minimization == False

        # TODO filter out non-diverse solutions
        if self.elitism:
            # Combine parents and offspring
            combined_population = parents + offspring
            # Sort by fitness
            combined_population.sort(key=lambda x: x.fitness, reverse=reverse)
            # Select the top individuals to form the new population
            new_population = combined_population[: self.n_parents]
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
        Main loop to evolve the solutions until the evolutionary budget is exhausted.
        The method iteratively refines solutions through interaction with the language model,
        evaluates their fitness, and updates the best solution found.

        Returns:
            tuple: A tuple containing the best solution and its fitness at the end of the evolutionary process.
        """
        # self.progress_bar = tqdm(total=self.budget)
        self.logevent("Initializing first population")
        self.initialize()  # Initialize a population
        # self.progress_bar.update(self.n_parents)

        if self.log:
            self.logger.log_population(self.population)

        self.logevent(
            f"Started evolutionary loop, best so far: {self.best_so_far.fitness}"
        )
        while len(self.run_history) < self.budget:
            # pick a new offspring population using random sampling
            new_offspring_population = np.random.choice(
                self.population, self.n_offspring, replace=True
            )

            new_population = []
            try:
                timeout = self.eval_timeout
                new_population_gen = Parallel(
                    n_jobs=self.max_workers,
                    timeout=timeout + 15,
                    backend="loky",
                    return_as="generator_unordered",
                )(
                    delayed(self.evolve_solution)(individual, i)
                    for i, individual in enumerate(new_offspring_population)
                )
            except Exception as e:
                print("Parallel time out .")

            for p in new_population_gen:
                self.run_history.append(p)
                new_population.append(p)
                print(f"Run history: {self.run_history}") # run history [1, 2, 3]
                print(f"New_population: {new_population}") # return 3, as 3 is the latest
            self.generation += 1

            if self.log:
                self.logger.log_population(new_population)

            # Update population and the best solution
            self.population = self.selection(self.population, new_population)
            self.update_best()
            self.logevent(
                f"Generation {self.generation}, best so far: {self.best_so_far.fitness}"
            )
            

        return self.best_so_far
