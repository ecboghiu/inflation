from sympy.combinatorics import Permutation

from inflation import InflationProblem, InflationLP
import numpy as np

Bell = InflationProblem(dag={"AB": ["A", "B"]},
                            outcomes_per_party=(2,2),
                            settings_per_party=(2,2),
                            inflation_level_per_source=(1,),
                            order=["A","B"],
                            classical_sources='all')

PR_box = np.zeros((2,2,2,2), dtype=float)
for a,b,x,y in np.ndindex(*PR_box.shape):
    if np.bitwise_xor(a,b) == np.bitwise_and(x,y):
        PR_box[a,b,x,y] = 0.5

(discovered_symmetries_lexorder, discovered_symmetries_original) = Bell.discover_distribution_symmetries(PR_box)

# from sympy.combinatorics.perm_groups import PermutationGroup, Permutation
# g = PermutationGroup([Permutation(g) for g in discovered_symmetries_original])
# print(g.strong_gens)

for sym in discovered_symmetries_original:
    interpreted_sym = Bell._interpret_original_symmetry(sym)
    new_sym = {str(orig): str(new) for (orig, new) in interpreted_sym.items() if not orig==new}
    if new_sym:
        print(new_sym)
Bell.lexorder_symmetries = Bell._incorporate_new_symmetries(discovered_symmetries_lexorder)
print("Group order: ", len(Bell.lexorder_symmetries))

BellLP = InflationLP(Bell, verbose=2, include_all_outcomes=False)
BellLP.set_distribution(PR_box, use_lpi_constraints=True)
print(BellLP.known_moments)
BellLP.solve()
print(BellLP.certificate_as_string())

#and the PR box is infeasible, all is good!