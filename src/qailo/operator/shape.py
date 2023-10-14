def shape(n, is_density_matrix=False):
    if is_density_matrix:
        return (2,) * (2 * n) + (1, 1)
    else:
        return (2,) * (2 * n)
