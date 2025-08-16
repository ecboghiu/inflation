# import sys
# sys.path.append('/Users/pedrolauand/inflation')
from inflation import InflationProblem, InflationLP, InflationSDP
import numpy as np
"""
Crazy idea: Use dummy intermediate latents to encode causal symmetry (to indicate the different Hilbert spaces of a single source)
"""
def exists_shared_source_modified(inf_indices1: np.ndarray,
                            inf_indices2: np.ndarray) -> bool:
    common_sources = np.logical_and(inf_indices1, inf_indices2)
    if not np.any(common_sources):
        return False
    return not set(inf_indices1[common_sources]).isdisjoint(set(inf_indices2[common_sources]))
def overlap_matrix(all_inflation_indxs: np.ndarray) -> np.ndarray:
    n = len(all_inflation_indxs)
    adj_mat = np.eye(n, dtype=bool)
    for i in range(1, n):
        inf_indices_i = all_inflation_indxs[i]
        for j in range(i):
            inf_indices_j = all_inflation_indxs[j]
            if exists_shared_source_modified(inf_indices_i, inf_indices_j):
                adj_mat[i, j] = True
    adj_mat = np.logical_or(adj_mat, adj_mat.T)
    return adj_mat
# def name_interpret_always_copy_indices(*args, **kwargs):
#     return InflationProblem._interpretation_to_name(*args, include_copy_indices=True)

def ring_problem(inflation_level: int, nof_outcomes: int = 2) -> InflationProblem:
    inf_prob = InflationProblem(
        dag={"i1": ["A"],
             "i2": ["A"], },
        outcomes_per_party=(nof_outcomes,),
        settings_per_party=(1,),
        classical_sources=None,
        inflation_level_per_source=(inflation_level,inflation_level),
        order=["A"],
        really_just_one_source=False)

    to_stabilize = np.flatnonzero(inf_prob._lexorder[:, 1] == inf_prob._lexorder[:, 2])


    #Fix factorization
    inf_prob._inflation_indices_overlap = overlap_matrix(inf_prob._all_unique_inflation_indices)

    # Fix symmetries
    new_symmetries = np.array([
        perm for perm in inf_prob.symmetries
        if np.array_equal(np.sort(perm[to_stabilize]), to_stabilize)
    ], dtype=int)
    inf_prob.symmetries = new_symmetries
    # inf_prob._interpretation_to_name = name_interpret_always_copy_indices

    return inf_prob


prob = ring_problem(4, 2)
# prob.add_symmetries(prob._setting_specific_outcome_relabelling_symmetries)
ring_SDP = InflationSDP(prob, verbose=2, include_all_outcomes=False)
ring_SDP.generate_relaxation("physical2")
# ring_SDP = InflationLP(prob, verbose=2)


print("Quantum inflation **nonfanout/commuting** factors:")
print(ring_SDP.physical_atoms)
# print(ring_SDP.atomic_factors)
# print("Quantum inflation **noncommuting** factors:")
# print(sorted(set(ring_4_SDP.atomic_factors).difference(ring_4_SDP.physical_atoms)))

#
# from inflation import max_within_feasible
# from sympy import Symbol
#
# v = Symbol("v")
# # v = 3/4 # EJM specifically, or 3/7 for local bound.
# # v = 1
# atomic_known_values_symbolic = {"1": 1}
# atomic_known_values_symbolic["P[A^{1,2}=0]"] = 1 / 4
# atomic_known_values_symbolic["P[A^{1,2}=0 A^{2,3}=0]"] = v / 8 + (1 - v) / 16
# atomic_known_values_symbolic["P[A^{1,2}=0 A^{2,3}=1]"] = v / 24 + (1 - v) / 16
# atomic_known_values_symbolic["P[A^{1,2}=0 A^{2,3}=0 A^{3,1}=0]"] = v / 8 + (1 - v) / 64
# atomic_known_values_symbolic["P[A^{1,2}=0 A^{2,3}=0 A^{3,1}=1]"] = (1 - v) / 64
# atomic_known_values_symbolic["P[A^{1,2}=0 A^{2,3}=1 A^{3,1}=2]"] = v / 48 + (1 - v) / 64
#
# # For optimization, we need to add the zero key.
# atomic_known_values_symbolic["0"] = 0
# # atomic_known_values_symbolic2 = {key: 768*val for key, val in atomic_known_values_symbolic.items()}
# print("Atomic known values:", atomic_known_values_symbolic)
# ring_SDP.update_values(atomic_known_values_symbolic)
# known_values_symbolic = ring_SDP.known_moments
# print("All known values:", known_values_symbolic)
#
# # ring_SDP.solve(solve_dual=True)
# max_vis, cert = max_within_feasible(ring_SDP, known_values_symbolic, "dual", precision=1e-8,
#                                     return_last_certificate=True,
#                                     verbose=True)
# print("Maximum visibility: ", max_vis)
# print("Certificate:", cert)