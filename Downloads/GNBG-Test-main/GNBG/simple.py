import logging
import os

os.makedirs('logging', exist_ok=True)  # Create the directory if it doesn't exist
# Configure the logging
logging.basicConfig(
    level=logging.INFO,                                # Set the minimum level of logs to capture
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    filename='logging/app.log',                                # File to write logs to
    filemode='a'                                       # Append mode ('w' for overwrite)
)

import os
import numpy as np
# from ioh import get_problem, logger
import re
from LLaMEA.llamea import LLaMEA, Gemini_LLM
from codes.gnbg_python.GNBG_instances import GNBG
from calc_aocc_from_gnbg import calculate_aocc_from_gnbg_history
from scipy.io import loadmat
from task_prompt import task_prompt_gnbg, simplified_task_prompt




if __name__ == "__main__":
    problem_idx_to_load = 1
    
    
    # Execution code starts here
    # AIzaSyAmHOlzt0LgKmgr2Mu2Fu7dEpE7PFDeNTs
    # AIzaSyCbiZx7Pmr5kWUihPXQ8nGvEsuo80kaiWE
    # api_key = os.getenv("GEMINI_API_KEY")
    api_key = "AIzaSyAmHOlzt0LgKmgr2Mu2Fu7dEpE7PFDeNTs"
    ai_model = "gemini-1.5-flash"
    experiment_name = "pop1-5"
    llm = Gemini_LLM(api_key, ai_model)
    
    def evaluateGNBG(solution, explogger=None, details=False): # we need to change this function to GNBG 
        auc_mean = 0
        auc_std = 0
        detailed_aucs = [0, 0, 0] # 3 function groups of GNBG benchmark, 3 detaiteed fedback       code = solution.code
        algorithm_name = solution.name
        code = solution.code
        print("Gonna execute")
        exec(code, globals()) # extract the code part inside the string, ex exec("a = 3 + 4") -> print(a) -> 7
        print("Finish execute")
        error = ""
        
        aucs = []
        detail_aucs = []
        algorithm = None
        for dim in [5]: # loading logic of BBOB, this is the dimension
            for fid in np.arange(1, 25): # cal 24 functions from GNBG
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
                problem = GNBG(MaxEvals, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate, CompMinPos, CompSigma, CompH, Mu, Omega, Lambda, RotationMatrix, OptimumValue, OptimumPosition)

              
                try:
                    algorithm = globals()[algorithm_name](budget=10, dim=problem.Dimension,
                        lower_bounds = [problem.MinCoordinate] * problem.Dimension,
                        upper_bounds = [problem.MaxCoordinate] * problem.Dimension
                        )
                    algorithm.optimize(objective_function=problem.fitness, acceptance_threshold = 1e-8)
                    print(f"Run on dim {dim}, function {fid}")
                    logging.info(f"Run on dim {dim}, function {fid}")
                    print(f"FeHistory: {problem.FEhistory}")
                    logging.info(f"FeHistory: {problem.FEhistory}")
                    print(f"Expected Optimum FE: {problem.OptimumValue}")
                    logging.info(f"Expected Optimum FE: {problem.OptimumValue}")
                except Exception:
                    print("Can not run the algorithm")
                    logging.error("Can not run the algorithm")
                    
                auc = calculate_aocc_from_gnbg_history(fe_history=problem.FEhistory,
                                                        optimum_value=problem.OptimumValue, 
                                                    budget_B=problem.MaxEvals
                                                )
                # fuck, it get low because MaxEvals is so large   
                aucs.append(auc)
                detail_aucs.append(auc)
                print(f'Aucs is: {aucs}')
                logging.info(f'Aucs is: {aucs}')
                print(f'Detail_aucs is: {detail_aucs}')
                logging.info(f'Detail_aucs is: {detail_aucs}')
                if fid == 6:
                    detailed_aucs[0] = np.mean(detail_aucs)
                    detail_aucs = []
                # group 2: multimodal instances with single component, 
                if fid == 15:
                    detailed_aucs[1] = np.mean(detail_aucs)
                    detail_aucs = []
                # group 3: multimodal instances with multiple component
                if fid == 24:
                    detailed_aucs[2] = np.mean(detail_aucs)
                    detail_aucs = []

        auc_mean = np.mean(aucs)
        auc_std = np.std(aucs)
        print(f'Auc_mean is: {auc_mean}')
        print(f'Auc_std is: {auc_std}')
      

        feedback = f"The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.2f} with standard deviation {auc_std:0.2f}."
        if details:
            feedback = (
                f"{feedback}\nThe mean AOCC score of the algorithm {algorithm_name} on Unimodal instances was {detailed_aucs[0]:.02f}, "
                f"on Multimodal instances with a single component {detailed_aucs[1]:.02f}, "
                f"on Multimodal instances with multiple components {detailed_aucs[2]:.02f}" 
            )

        print(algorithm_name, algorithm, auc_mean, auc_std)
        solution.add_metadata("aucs", aucs)
        solution.set_scores(auc_mean, feedback)

        return solution

    for experiment_i in [10]:
        # A 1+1 strategy
        es = LLaMEA(
            evaluateGNBG,
            n_parents=1,
            n_offspring=1,
            llm=llm,
            task_prompt=simplified_task_prompt,
            experiment_name=experiment_name,
            elitism=True,
            HPO=False,
            budget=10,
            log=False, 
            adaptive_mutation=True, 
            minimization=True
        )
        print(es.run())