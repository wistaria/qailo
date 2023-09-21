import qailo as q

v = q.sv.zero(3)
print(v)
print(q.sv.vector(v))
print(q.sv.probability(v))

v = q.sv.multiply(q.op.h(), v, [0])
print("1", q.sv.vector(v))
v = q.sv.multiply(q.op.h(), v, [2])
print("2", q.sv.vector(v))
v = q.sv.multiply(q.op.cx(), v, [0, 1])
print("3", q.sv.vector(v))
v = q.sv.multiply(q.op.cz(), v, [1, 2])
print("4", q.sv.vector(v))
v = q.sv.multiply(q.op.h(), v, [2])
print("5", q.sv.vector(v))

print(v)
print(q.sv.vector(v))
print(q.sv.probability(v))
