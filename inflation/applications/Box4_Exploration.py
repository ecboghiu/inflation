import numpy as np
from inflation import InflationProblem, InflationLP

# We want a triangle with settings
# Outcomes have cardinality 2
# Settings have cardinality 2
# 1 classical source
# 2 nonclassical sources
# inflation based taking copies of the nonclassical sources

triangle_one_classical_source_only_nonclassical_multiple_copies = InflationProblem(
    dag={
    "lambda_ABC": ["A", "B", "C"],
    "omega_AC": ["A", "C"],
    "omega_BC": ["B", "C"],
    "omega_AB": ["A", "B"]},
    order=['A', 'B', 'C'],
    outcomes_per_party=[2, 2, 2],
    settings_per_party=[2, 2, 2],
    inflation_level_per_source=[1, 2, 2, 2],
    classical_sources=["lambda_ABC"],
    verbose=2)

our_problem=InflationLP(triangle_one_classical_source_only_nonclassical_multiple_copies,
                        nonfanout=True)

box4 = np.ones((2,2,2,2,2,2))
for (a,b) in np.ndindex(2,2):
    if a != b:
        box4[a,b,:,0,1,:] = 0
for (b,c) in np.ndindex(2,2):
    if b != c:
        box4[:,b,c,:,0,1] = 0
for (a,c) in np.ndindex(2,2):
    if a != c:
        box4[a,:,c,1,:,0] = 0
for (a,b,c) in np.ndindex(2,2,2):
    if np.mod(a + b + c,2) == 1:
        box4[a,b,c,0,0,0] = 0
    else:
        box4[a, b, c, 1, 1, 1] = 0

# for (x,y,z) in np.ndindex(2,2,2):
#     print(box4[:,:,:,x,y,z])

# box4 = box4/4
our_problem.set_distribution(box4)
our_problem.solve(dualise=False)
print(f"Objective value: {our_problem.objective_value}")
print(our_problem.certificate_as_string())



