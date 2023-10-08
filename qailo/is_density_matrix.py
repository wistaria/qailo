def is_density_matrix(dm):
    return dm.shape[-1] == 1 and dm.shape[-2] == 1 and dm.shape[-3] == 2
