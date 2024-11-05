from inflation import InflationProblem, InflationLP
import numpy as np


d=3

triangle = InflationProblem(dag={"AB": ["A", "B"],
                                 "BC": ["B", "C"],
                                 "AC": ["A", "C"]},
                            outcomes_per_party=(d,d,d),
                            settings_per_party=(1,1,1),
                            inflation_level_per_source=(2,2,2),
                            order=["A","B","C"],
                            classical_sources=None)
# print(triangle._lexorder)
permuting_copy_indices = triangle._lexorder[:,[0,3,1,2,4,5]]
permuting_copy_indices[permuting_copy_indices[:,2]==0,0] = 1
permuting_copy_indices[permuting_copy_indices[:,3]==0,0] = 2
permuting_copy_indices[permuting_copy_indices[:,1]==0,0] = 3
# print(np.stack((triangle._lexorder,permuting_copy_indices),axis=1))
new_lexorder_permutation = np.array([triangle._lexorder_lookup[op.tobytes()] for op in permuting_copy_indices])
# print(triangle.lexorder_symmetries)
all_symmetries_new = np.vstack((triangle.lexorder_symmetries,
                                triangle.lexorder_symmetries[:,new_lexorder_permutation],
                                triangle.lexorder_symmetries[:,new_lexorder_permutation[new_lexorder_permutation]]
                                ))
all_symmetries_new_v2 = np.vstack((triangle.lexorder_symmetries,
                                new_lexorder_permutation[triangle.lexorder_symmetries],
                                new_lexorder_permutation[new_lexorder_permutation][triangle.lexorder_symmetries]
                                ))
triangle.lexorder_symmetries = np.unique(all_symmetries_new,axis=0)
assert np.array_equal(np.unique(all_symmetries_new,axis=0), np.unique(all_symmetries_new_v2,axis=0)), "Implementation matters?!"
assert len(triangle.lexorder_symmetries)==len(all_symmetries_new), "Somehow a generator showed up twice!"
print(triangle.lexorder_symmetries)
identity_sym = triangle.lexorder_symmetries[0]
for sym in triangle.lexorder_symmetries:
    assert np.array_equal(np.unique(sym), identity_sym), "Symmetry is not a permutation!"
# print(triangle._lexrepr_to_names)

import itertools
victors_distribution = np.zeros(shape=(d,d,d,1,1,1), dtype=float)
for a in range(d):
    for b in range(0,a):
        for c in range(0,b):
            val = (1-1/d)/(d*(d-1)*(d-2))
            indices_choices = itertools.permutations((a,b,c))
            for indices in indices_choices:
                victors_distribution[indices] = val
    victors_distribution[a,a,a] = 1/d*1/d
print(np.sum(victors_distribution))
print(len(np.flatnonzero(victors_distribution)))

triangleLP = InflationLP(triangle, verbose=2)
triangleLP.set_distribution(victors_distribution, use_lpi_constraints=True)
triangleLP.solve()
print(triangleLP.status)