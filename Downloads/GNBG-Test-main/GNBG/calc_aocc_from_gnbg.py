import numpy as np
def calculate_aocc_from_gnbg_history(fe_history, optimum_value, budget_B, 
                                     log_error_lower_bound=-8.0,  # Corresponds to 10^-8 error
                                     log_error_upper_bound=2.0):   # Corresponds to 10^2 error
    """
    Calculates Area Over the Convergence Curve (AOCC) from GNBG FEhistory.
    Higher AOCC is better (1.0 is optimal).
    """
    if len(fe_history) == 0:
        print(f"Length of fe_history is 0, aocc is 0")
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