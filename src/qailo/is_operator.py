def is_operator(op):
    for x in op.shape:
        if x != 2 and x != 4:
            return False
    return len(op.shape) % 2 == 0
