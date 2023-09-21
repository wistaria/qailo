import qailo as q

v = q.sv.zero(3)
print(v)
print(q.sv.vector(v))
print(q.sv.probability(v))

v = q.sv.multiply(q.op.h(), v, [0])
v = q.sv.multiply(q.op.h(), v, [2])
v = q.sv.multiply(q.op.cx(), v, [0, 1])
v = q.sv.multiply(q.op.cz(), v, [1, 2])
v = q.sv.multiply(q.op.h(), v, [2])

print(v)
print(q.sv.vector(v))
print(q.sv.probability(v))
