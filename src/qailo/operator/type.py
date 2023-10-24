def is_density_matrix(q):
    return (
        q.shape[-1] == 1
        and q.shape[-2] == 1
        and q.shape[-3] > 1
        and len(q.shape) % 2 == 0
    )


def is_operator(q):
    return q.shape[-1] > 1 and len(q.shape) % 2 == 0


def num_qubits(q):
    if is_density_matrix(q):
        return (len(q.shape) - 2) // 2
    elif is_operator(q):
        return len(q.shape) // 2
