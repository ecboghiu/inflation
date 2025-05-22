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
def name_interpret_always_copy_indices(*args, **kwargs):
    return InflationProblem._interpretation_to_name(*args, include_copy_indices=True)

def ring_problem(n: int) -> InflationProblem:
    inf_prob = InflationProblem(
        dag={"i1": ["A"],
             "i2": ["A"], },
        outcomes_per_party=(2,),
        settings_per_party=(1,),
        classical_sources=None,
        inflation_level_per_source=(n,n),
        order=["A"])

    to_stabilize = np.flatnonzero(inf_prob._lexorder[:, 1] == inf_prob._lexorder[:, 2])

    #Artificially kill all self-loops
    # inf_prob._default_notcomm[:, to_stabilize] = True
    # inf_prob._default_notcomm[to_stabilize] = True
    # inf_prob._default_notcomm[np.ix_(to_stabilize, to_stabilize)] = False

    #Fix factorization
    inf_prob._inflation_indices_overlap = overlap_matrix(inf_prob._all_unique_inflation_indices)

    # Fix symmetries
    # print("Never use: ", to_stabilize)
    new_symmetries = np.array([
        perm for perm in inf_prob.symmetries
        if np.array_equal(np.sort(perm[to_stabilize]), to_stabilize)
    ], dtype=int)
    inf_prob.symmetries = new_symmetries



    #Hacks to prevent knowability assumptions
    # inf_prob.is_network = False
    # inf_prob._is_knowable_q_non_networks = (lambda x: False)

    inf_prob._interpretation_to_name = name_interpret_always_copy_indices

    return inf_prob
# def ring_LP(n: int, **kwargs) -> InflationLP:
#     inf_lp = InflationLP(ring_problem(n), **kwargs)
#     inf_lp.all_commuting_q_1d = (lambda x: False)
#     inf_lp.all_commuting_q_2d = (lambda x: False)
#     inf_lp.all_operators_commute = False
#     return inf_lp

prob_4 = ring_problem(4)
# print(prob_4._compatible_template_measurements.astype(int))
# cliques = prob_4.all_and_maximal_compatible_templates()[-1]
# for clique in cliques[:4]:
#     print(prob_4._compatible_template_measurements.astype(int)[np.ix_(clique,clique)])
#     print(prob_4._lexorder[prob_4._template_idxs[clique]])

ring_4_LP = InflationLP(prob_4, verbose=2)
print("Nonfanout inflation atomic factors:")
print(ring_4_LP.atomic_factors)

ring_4_SDP = InflationSDP(prob_4, verbose=2)
ring_4_SDP.generate_relaxation("npa2")

print("Quantum inflation **nonfanout/commuting** factors:")
print(ring_4_SDP.physical_atoms)
print("Quantum inflation **noncommuting** factors:")
print(sorted(set(ring_4_SDP.atomic_factors).difference(ring_4_SDP.physical_atoms)))
# # print(ring_4_SDP.momentmatrix)
# #
# known_values = {}
# E1 = 0
# # E1 ≥0.1656
# E2 = -1 / np.sqrt(2)
# # Inputting values
# known_values["P[A^{1,2}=0]"] = 1 / 2 * (1 + E1)
#
# known_values["P[A^{1,2}=0 A^{2,3}=0]"] = 1 / 4 * (1 + 2*E1 + E2)
# print(known_values)
#
# ring_4_SDP.update_values(known_values)
# ring_4_SDP.solve(solve_dual=False)
#
