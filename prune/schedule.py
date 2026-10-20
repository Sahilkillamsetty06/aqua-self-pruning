def cubic_sparsity(step, total_steps, final_sparsity, initial_sparsity=0.0):
    """Cubic ramp: s(t)=s_f + (s_i-s_f)(1-t/T)^3."""
    if total_steps <= 0: return final_sparsity
    t=min(max(step, 0), total_steps)/total_steps
    return final_sparsity + (initial_sparsity-final_sparsity)*(1-t)**3
