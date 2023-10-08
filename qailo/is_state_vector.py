def is_state_vector(v):
    return v.shape[-1] == 1 and v.shape[-2] == 2
