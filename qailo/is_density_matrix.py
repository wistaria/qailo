def is_density_matrix(dm):
    if dm.shape[-1] == 1 and dm.shape[-2] == 1 and dm.shape[-3] == 2:
        n = (len(dm.shape) - 2) // 2
        assert dm.shape == (2,) * (2 * n) + (1, 1)
        return True
    else:
        return False
