def is_state_vector(sv):
    if sv.shape[-1] == 1 and sv.shape[-2] == 2:
        n = len(sv.shape) - 1
        assert sv.shape == (2,) * n + (1,)
        return True
    else:
        return False
