import numpy as np
import re
import textwrap
def calculate_aocc_from_gnbg_history(fe_history, optimum_value, budget_B, 
                                     log_error_lower_bound=-8.0,  # Corresponds to 10^-8 error
                                     log_error_upper_bound=2.0, early_stop = False):   # Corresponds to 10^2 error
    # To evaluate this, we consider the fe-history length up to budget B
    """
    Calculates Area Over the Convergence Curve (AOCC) from GNBG FEhistory.
    Higher AOCC is better (1.0 is optimal).
    """
    if len(fe_history) == 0:
        print(f"Length of fe_history is 0, aocc is 0")
        return 0.0 # No evaluations, worst AOCC

    actual_evals = len(fe_history) # should be equal to budget
    
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
        print("When calculating aocc, acutal evaluation < budget B")
        best_error_history_at_fe.extend([best_error_so_far] * (budget_B - actual_evals))
    else:    
        print("When calculating aocc, acutal evaluation >= budget B")

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

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()
    
def extract_to_hs(input_string: str):
    code_blocks = input_string.split("```python\n")[1:]

    try:
        parameter_ranges_block = "import numpy as np\n" + code_blocks[1].split("```")[0].strip()
        if any(keyword in parameter_ranges_block for keyword in ['inf', 'np.inf', 'None']):
            return None, None
        exec_globals = {}
        exec(parameter_ranges_block, exec_globals)
        parameter_ranges = exec_globals['parameter_ranges']
    except:
        return None, None

    function_block = code_blocks[0].split("```")[0].strip()

    paren_count = 0
    in_signature = False
    signature_start_index = None
    signature_end_index = None

    # Loop through the function block to find the start and end of the function signature
    for i, char in enumerate(function_block):
        if char == "d" and function_block[i:i + 3] == 'def':
            in_signature = True
            signature_start_index = i
        if in_signature:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            if char == ':' and paren_count == 0:
                signature_end_index = i
                break

    if signature_start_index is not None and signature_end_index is not None:
        function_signature = function_block[signature_start_index:signature_end_index + 1]
        for param in parameter_ranges:
            pattern = rf"(\b{param}\b[^=]*=)[^,)]+"
            replacement = r"\1 {" + param + "}"
            function_signature = re.sub(pattern, replacement, function_signature, flags=re.DOTALL)
        function_block = function_block[:signature_start_index] + function_signature + function_block[
                                                                                       signature_end_index + 1:]

    return parameter_ranges, function_block

def extract_class_name_and_code(code_string: str):
    cleaned_code = code_string.encode().decode('unicode_escape')

    class_name_match = re.search(r'class\s+(\w+)\s*[:\(]', cleaned_code)
    class_name = class_name_match.group(1) if class_name_match else None

    cleaned_code = textwrap.dedent(cleaned_code)

    return class_name, cleaned_code