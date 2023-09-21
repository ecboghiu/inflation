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
    "omega_AB": ["A", "B"],
    "omega_AC": ["A", "C"],
    "omega_AD": ["A", "D"],
    "omega_AE": ["A", "E"],
    "omega_BC": ["B", "C"],
    "omega_BD": ["B", "D"],
    "omega_BE": ["B", "E"],
    "omega_CD": ["C", "D"],
    "omega_CE": ["C", "E"],
    "omega_DE": ["D", "E"]},
    order=['A', 'B', 'C', 'D', 'E'],
    outcomes_per_party=[2, 2, 2, 2, 2],
    settings_per_party=[2, 2, 2, 2, 2],
    inflation_level_per_source=([2]*0 + [1]*10),
    classical_sources='all',
    verbose=2)

our_problem=InflationLP(triangle_one_classical_source_only_nonclassical_multiple_copies,
                        nonfanout=False)

box5 = np.ones((2,2,2,2,2,2,2,2,2,2))
for (a,b,c) in np.ndindex(2,2,2):
    if np.mod(a + b + c,2) == 1:
        box5[a,b,c,:,:,0,1,0,:,:] = 0
for (b,c,d) in np.ndindex(2,2,2):
    if np.mod(b + c + d,2) == 1:
        box5[:,b,c,d,:,:,0,1,0,:] = 0
for (c,d,e) in np.ndindex(2,2,2):
    if np.mod(c + d + e,2) == 1:
        box5[:,:,c,d,e,:,:,0,1,0] = 0
for (a,d,e) in np.ndindex(2,2,2):
    if np.mod(a + d + e,2) == 1:
        box5[a,:,:,d,e,0,:,:,0,1] = 0
for (a,b,e) in np.ndindex(2,2,2):
    if np.mod(a + b + e,2) == 1:
        box5[a,b,:,:,e,1,0,:,:,0] = 0
for (a,b,c,d,e) in np.ndindex(2,2,2,2,2):
    if np.mod(a + b + c + d + e,2) == 0:
        box5[a,b,c,d,e,1,1,1,1,1] = 0

for (a,b,c,d,e, x,y,z,w,q) in np.ndindex(2,2,2,2,2,2,2,2,2,2):
    both = box5[a,b,c,d,:, x,y,z,w,0] + box5[a,b,c,d,:, x,y,z,w,1]
    box5[a, b, c, d, :, x, y, z, w, 0] = both/2
    box5[a, b, c, d, :, x, y, z, w, 1] = both / 2
    both = box5[a, b, c, :, e, x, y, z, 0, q] + box5[a, b, c, :, e, x, y, z, 1, q]
    box5[a, b, c, :, e, x, y, z, 0, q] = both / 2
    box5[a, b, c, :, e, x, y, z, 1, q] = both / 2
    both = box5[a, b, :, d, e, x, y, 0, w, q] + box5[a, b, :, d, e, x, y, 1, w,
                                                q]
    box5[a, b, :, d, e, x, y, z, 0, q] = both / 2
    box5[a, b, :, d, e, x, y, z, 1, q] = both / 2
    both = box5[a, :, c, d, e, x, 0, z, w, q] + box5[a, :, c, d, e, x, 1, z, w,
                                                q]
    box5[a, :, c, d, e, x, 0, z, w, q] = both / 2
    box5[a, :, c, d, e, x, 1, z, w, q] = both / 2
    both = box5[:, b, c, d, e, 0, y, z, w, q] + box5[:, b, c, d, e, x, 1, z, w,
                                                q]
    box5[:, b, c, d, e, 0, y, z, w, q] = both / 2
    box5[:, b, c, d, e, 1, y, z, w, q] = both / 2

for (a,b,c) in np.ndindex(2,2,2):
    if np.mod(a + b + c,2) == 1:
        box5[a,b,c,:,:,0,1,0,:,:] = 0
for (b,c,d) in np.ndindex(2,2,2):
    if np.mod(b + c + d,2) == 1:
        box5[:,b,c,d,:,:,0,1,0,:] = 0
for (c,d,e) in np.ndindex(2,2,2):
    if np.mod(c + d + e,2) == 1:
        box5[:,:,c,d,e,:,:,0,1,0] = 0
for (a,d,e) in np.ndindex(2,2,2):
    if np.mod(a + d + e,2) == 1:
        box5[a,:,:,d,e,0,:,:,0,1] = 0
for (a,b,e) in np.ndindex(2,2,2):
    if np.mod(a + b + e,2) == 1:
        box5[a,b,:,:,e,1,0,:,:,0] = 0
for (a,b,c,d,e) in np.ndindex(2,2,2,2,2):
    if np.mod(a + b + c + d + e,2) == 0:
        box5[a,b,c,d,e,1,1,1,1,1] = 0




norm_factors = np.ones((2,2,2,2,2));
for (x,y,z,w,q) in np.ndindex(2,2,2,2,2):
    norm_factors[x,y,z,w,q] = box5[:,:,:,:,:,x,y,z,w,q].sum()
for (a,b,c,d,e, x,y,z,w,q) in np.ndindex(2,2,2,2,2,2,2,2,2,2):
    box5[a,b,c,d,e, x,y,z,w,q] = box5[a,b,c,d,e, x,y,z,w,q]/norm_factors[x,y,z,w,q]
# print(box5[:,:,:,:,:,0,0,0,0,1])

our_problem.set_distribution(box5, shared_randomness=True)
our_problem.solve(dualise=False)
# print(f"Objective value: {our_problem.objective_value}")
print(our_problem.certificate_as_string())



