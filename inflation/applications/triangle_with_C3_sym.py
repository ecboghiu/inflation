from inflation import InflationProblem, InflationLP, InflationSDP
import numpy as np

d=3

triangle = InflationProblem(dag={"AB": ["A", "B"],
                                 "BC": ["B", "C"],
                                 "AC": ["A", "C"]},
                            outcomes_per_party=(d,d,d),
                            settings_per_party=(1,1,1),
                            inflation_level_per_source=(2,2,2),
                            order=["A","B","C"],
                            classical_sources='all')
# print(triangle._lexorder)
permuting_copy_indices = triangle._lexorder[:,[0,3,1,2,4,5]]
permuting_copy_indices[permuting_copy_indices[:,2]==0,0] = 1
permuting_copy_indices[permuting_copy_indices[:,3]==0,0] = 2
permuting_copy_indices[permuting_copy_indices[:,1]==0,0] = 3
new_lexorder_permutation_sources = np.array([triangle._lexorder_lookup[op.tobytes()] for op in permuting_copy_indices])


permuting_outcome_labels = triangle._lexorder.copy()
for i in range(d):
    permuting_outcome_labels[triangle._lexorder[:,-1] == i,-1] = np.mod(i+1,d)
new_lexorder_cycle_outcomes = np.array([triangle._lexorder_lookup[op.tobytes()] for op in permuting_outcome_labels])

permuting_outcome_labels = triangle._lexorder.copy()
for i in range(2):
    permuting_outcome_labels[triangle._lexorder[:,-1] == i,-1] = np.mod(i+1,2)
new_lexorder_swap_outcomes = np.array([triangle._lexorder_lookup[op.tobytes()] for op in permuting_outcome_labels])

from sympy.combinatorics import Permutation, PermutationGroup

group_generators = [Permutation(perm) for perm in triangle.lexorder_symmetries]
group_generators.extend([Permutation(perm) for perm in [new_lexorder_permutation_sources,
                                                        new_lexorder_cycle_outcomes,
                                                        new_lexorder_swap_outcomes
                                                        ]])

all_symmetries_new = np.array(list(PermutationGroup(group_generators).generate_schreier_sims(af=True)))
print("Group order: ", len(all_symmetries_new))

triangle.lexorder_symmetries = np.unique(all_symmetries_new,axis=0)
# assert len(triangle.lexorder_symmetries)==len(all_symmetries_new), "Somehow a generator showed up twice!"
# print(triangle.lexorder_symmetries)
# identity_sym = triangle.lexorder_symmetries[0]
# for sym in triangle.lexorder_symmetries:
#     assert np.array_equal(np.unique(sym), identity_sym), "Symmetry is not a permutation!"


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
# print(np.sum(victors_distribution))
# print(len(np.flatnonzero(victors_distribution)))

triangleLP = InflationLP(triangle, verbose=2, supports_problem=True)
# triangleLP.set_distribution(victors_distribution, use_lpi_constraints=True)
for monomial in triangleLP.atomic_monomials:
    if monomial.n_operators < 3:
        print(monomial)

triangleLP.set_bounds({
    "P[A=0]": 1,
    # "P[A=1]": 1,
    # "P[A=2]": 1
}, bound_type="lo")
triangleLP.update_values({
    "P[A=0 B=0]": 0,
    # "P[A=1 B=1]": 0,
    # "P[A=2 B=2]": 0
}, use_lpi_constraints=True)
triangleLP.solve()
print(triangleLP.status)
solution_x = triangleLP.solution_object["x"]
print("These should be zero:")
print(solution_x["P[A=0 B=0]"])
print(solution_x["P[A=0 B=0 C=0]"])
print(solution_x["P[A=0 B=0 C=1]"])
print("These should be greater than 1:")
print(solution_x["P[A=0]"])
print(solution_x["P[A=0 B=1]"])
# print(solution_x["P[A=0 B=2]"])
# print(solution_x["P[A=1 B=2]"])
# print(solution_x["P[A=0 C=1]"])
# print(solution_x["P[A=0 C=1]"])
# print(solution_x["P[A=1 C=2]"])
print(solution_x["P[A=0 B=1 C=2]"])



# triangleSDP = InflationSDP(triangle, verbose=2, include_all_outcomes=False, supports_problem=True)
# triangleSDP.generate_relaxation("local1")
# triangleSDP.set_distribution(victors_distribution, use_lpi_constraints=True)
# triangleSDP.solve()
# print(triangleSDP.status)

# for moment in triangleSDP.atomic_monomials:
#     if moment.n_operators < 3:
#         print(moment)
#
# triangleSDP.set_bounds({
#     "P[A=0]": 1,
#     "P[A=1]": 1,
#     "P[A=2]": 1}, bound_type="lo")
# triangleSDP.update_values({
#     "P[A=0 B=0]": 0,
#     "P[A=1 B=1]": 0,
#     "P[A=2 B=2]": 0
# }, use_lpi_constraints=True)
# triangleSDP.solve()