import logging
import os
import numpy as np
import re
from llamea import Gemini_LLM
# from llamea_with_hs import LLaMEA
from llamea_with_hs_single_func import LLaMEA
from codes.gnbg_python.GNBG_instances import GNBG
from my_utils.utils import * # include calculate aocc
from scipy.io import loadmat
    
# Execution code starts here
# AIzaSyAmHOlzt0LgKmgr2Mu2Fu7dEpE7PFDeNTs
# AIzaSyCbiZx7Pmr5kWUihPXQ8nGvEsuo80kaiWE
api_keys = [
        'AIzaSyARJfdVOsI9AKUK6gxvUszL_bn5Z_lr5Wg',
        'AIzaSyAS4WrMPg7WNPXLpCYIOG49cIYOL7vTAl8',
        'AIzaSyCK6miE77n6z7PUf0RNgj8seMiiVET-wqk'
]
# api_key = "AIzaSyAmHOlzt0LgKmgr2Mu2Fu7dEpE7PFDeNTs"
ai_model = "gemini-2.0-flash"
experiment_name = "pop1-5"

# llm = Gemini_LLM("AIzaSyCbA5uIXRIIWnTv7HCUGX75WoYJ4PeZWd0", ai_model)
llm_instances = [Gemini_LLM(key, ai_model) for key in api_keys]
# llm = Ollama_LLM(model="llama3.2:3b-instruct-fp16")
def evaluateGNBG(solution, explogger=None, details=True): # we need to change this function to GNBG 
    logger = logging.getLogger(__name__)
    auc_mean = 0
    auc_std = 0
    worst_aocc_score = 0.0 # Higher AOCC is better
    all_run_aocc_scores = []
    unimodal_aoccs, multimodal_single_aoccs, multimodal_multi_aoccs = [], [], []
    algorithm_name = solution.name
    code = solution.code
    exec(code, globals()) # extract the code part inside the string, ex exec("a = 3 + 4") -> print(a) -> 7
    error = ""
    
    aucs = []
    detail_aucs = []
    algorithm = None
    problem_indices_to_test = [16, 18, 19] 
    # 15, 24
    for fid in problem_indices_to_test: # cal 24 functions from GNBG
        # representative of 3 function group
        filename = f'f{fid}.mat'
        GNBG_tmp = loadmat(os.path.join("codes/gnbg_python", filename))['GNBG']
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
        problem = GNBG(150000, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate, CompMinPos, CompSigma, CompH, Mu, Omega, Lambda, RotationMatrix, OptimumValue, OptimumPosition)
        # max eval is just 1000
        # len of FEhistory should be equal to MaxEval or budget
        log_content = (
        f"--- GNBG Problem Parameters for f{fid} ---\n"
        f"  Dimension: {Dimension}, MaxEvals: {MaxEvals}\n"
        f"  Search Bounds: [{MinCoordinate}, {MaxCoordinate}]\n"
        f"  Number of Components: {CompNum}\n"
        f"  Known Optimum Value: {OptimumValue:.6f}\n"
        f"  Lambda (Curvature): {Lambda.flatten()}\n"
        f"  Mu (Asymmetry/Depth): {Mu.flatten()}\n"
        f"----------------------------------------"
        )
        
        logger.info(log_content)
        try:
            algorithm = globals()[algorithm_name](budget=problem.MaxEvals, dim=problem.Dimension,
                lower_bounds = [problem.MinCoordinate] * problem.Dimension,
                upper_bounds = [problem.MaxCoordinate] * problem.Dimension
                )
            algorithm.optimize(objective_function=problem.fitness, acceptance_threshold = 1e-8)
            # After this we get Fe History len up to budget = 10000
            print(f"Run on function {fid}")
            
        except Exception:
            print("Can not run the algorithm")
            logging.error("Can not run the algorithm")
            
        current_run_aocc = calculate_aocc_from_gnbg_history(fe_history=problem.FEhistory,
                                                optimum_value=problem.OptimumValue, 
                                            budget_B=problem.MaxEvals)
        
        logging.info(f"Run function {fid} complete. FEHistory len: {len(problem.FEhistory)}, AOCC: {current_run_aocc:.4f}")
        logging.info(f"FeHistory: {problem.FEhistory}")
        logging.info(f"Expected Optimum FE: {problem.OptimumValue}")
        if current_run_aocc >= 0.1:
            logging.info(f"Good algorithm:\nAlgorithm Name: {algorithm_name}\n{code}")
        # budget of auc and algorithm must match 
            
        # aucs.append(auc)
        # detail_aucs.append(auc)
        
        
        all_run_aocc_scores.append(current_run_aocc)
        if 1 <= fid <= 6: unimodal_aoccs.append(current_run_aocc)
        if 7 <= fid <= 15: multimodal_single_aoccs.append(current_run_aocc)
        if 16 <= fid <= 24: multimodal_multi_aoccs.append(current_run_aocc)
    
    unimodal_means = np.mean(unimodal_aoccs)
    multimodal_single_means = np.mean(multimodal_single_aoccs)
    multimodal_multi_means = np.mean(multimodal_multi_aoccs)
    
    logging.info(f"Unimodal AOCC mean: {unimodal_means:.4f}")
    logging.info(f"Multimodal (single component) AOCC mean: {multimodal_single_means:.4f}")
    logging.info(f"Multimodal (multiple components) AOCC mean: {multimodal_multi_means:.4f}")

    auc_mean = np.mean(all_run_aocc_scores)
    auc_std = np.std(all_run_aocc_scores)
    print(f'Auc_mean is: {auc_mean}')
    print(f'Auc_std is: {auc_std}')
    
    feedback = f"The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.2f} with standard deviation {auc_std:0.2f}."
    if details:
        feedback = (
            f"{feedback}\nThe mean AOCC score of the algorithm {algorithm_name} on Unimodal instances was {unimodal_means:.02f}, "
            f"The mean AOCC score of the algorithm {algorithm_name} on Multimodal instances with a single component {multimodal_single_means:.02f}, "
            f"The mean AOCC score of the algorithm {algorithm_name} on Multimodal instances with multiple components {multimodal_multi_means:.02f}" 
        )

    print(algorithm_name, algorithm, auc_mean, auc_std)
    solution.add_metadata("aucs", aucs)
    weights = {
    'unimodal': 0.1,        # f1-f6
    'multimodal_single': 0.2, # f7-f15
    'multimodal_multi': 0.7   # f16-f24 (The most important group)
    }
    weighed_aoccs = weights['unimodal'] * unimodal_means + weights['multimodal_single'] * multimodal_single_means \
        + weights['multimodal_multi'] * multimodal_multi_means
    # solution.set_scores(weighed_aoccs, feedback, aocc1 = unimodal_means, aocc2 = multimodal_single_means, aocc3 = multimodal_multi_means)
    solution.set_scores(auc_mean, feedback, aocc1 = unimodal_means, aocc2 = multimodal_single_means, aocc3 = multimodal_multi_means)
    logging.info(f"AOCC mean: {auc_mean:.4f}")
    logging.info(f"Weighed AOCC mean: {weighed_aoccs:.4f}")

    return solution
if __name__ == "__main__":
    for experiment_i in [1]:
    # A 1+1 strategy
        es = LLaMEA(
            evaluateGNBG,
            llms=llm_instances,
            experiment_name=experiment_name,
            budget=100,
        )
        print(es.run())