from collections import defaultdict

import numpy as np
from numpy import dtype

from inflation import InflationProblem, InflationSDP, InflationLP

# Although C has 4 settings, we only refer to one in the objective, so we'll lie WLOG in the input.
Bell_ABCD = InflationProblem(
    dag={"rho_ABC": ["A", "B", "C", "D"]},
    outcomes_per_party=[2, 2, 2, 2],
    settings_per_party=[2, 2, 1, 1],
    inflation_level_per_source=(1,),
    order=['A', 'B', 'C', 'D'],
    verbose=0,
    classical_sources=None)

print(Bell_ABCD._lexrepr_to_dicts)

setting_decoder = [(0,0),(1,1)]

print("Before manually adjusting noncommutation relations:")
print(Bell_ABCD._default_notcomm.astype(int))
n_ops = Bell_ABCD._nr_operators
# If we want to impose that when B and C have different settings they do not commute:
for i, j in np.ndindex((n_ops,n_ops)):
    op_i = Bell_ABCD._lexrepr_to_dicts[i]
    op_j = Bell_ABCD._lexrepr_to_dicts[j]
    if (op_i['Party'] == 'A'):
        (x1, x2) = setting_decoder[op_i['Composite Setting']]
        if (op_j['Party'] == 'C') and x1 > 0:
            Bell_ABCD._default_notcomm[i, j] = True
            Bell_ABCD._default_notcomm[j, i] = True
        if (op_j['Party'] == 'D') and x2 > 0:
            Bell_ABCD._default_notcomm[i, j] = True
            Bell_ABCD._default_notcomm[j, i] = True
print("After manually adjusting noncommutation relations:")
print(Bell_ABCD._default_notcomm.astype(int))

# Let the distribution be such that Charlie guesses Alice's outcome with perfect certainty
# v = 1 / np.sqrt(2)
# P_CharlieGuessesAlice = np.full((2,2,2,2,2,4), fill_value=np.nan)
# for a,b,c,x,y in np.ndindex(2,2,2,2,2):
#     z = 2*x + y
#     if a != c:
#         P_CharlieGuessesAlice[a, b, c, x, y, z] = 0
#     else:
#         P_CharlieGuessesAlice[a, b, c, x, y, z] = (1 / 4) * ((1 + (v) * ((-1) ** (a + b + x * y))))

# w = np.sqrt(2)/2-0.5
w = 1
CH_objective = {
    "P[A_0=0]": -1,
    "P[B_0=0]": -1,
    "P[A_0=0 B_0=0]": 1,
    "P[A_0=0 B_1=0]": 1,
    "P[A_1=0 B_0=0]": 1,
    "P[A_1=0 B_1=0]": -1}
TripartiteSameGame_objective = {
    "P[A_0=0]": -1,
    "P[C=0]": -1,
    "P[D=0]": -1,
    "P[A_0=0 C=0]": 1,
    "P[A_0=0 D=0]": 1,
    "P[C=0 D=0]": 1,
    "P[A_0=0 C=0 D=0]": 0}
ACSameGame_objective = {
    "P[A_0=0]": -1,
    "P[C=0]": -1,
    "P[A_0=0 C=0]": 2}
ADSameGame_objective = {
    "P[A_0=0]": -1,
    "P[D=0]": -1,
    "P[A_0=0 D=0]": 2}
monogamy_objective = defaultdict(int)
for k,v in CH_objective.items():
    monogamy_objective[k] += v
for k,v in TripartiteSameGame_objective.items():
    monogamy_objective[k] += v
# for k,v in ACSameGame_objective.items():
#     monogamy_objective[k] += v
# for k,v in ADSameGame_objective.items():
#     monogamy_objective[k] += v


one_intermediate_latent_SDP = InflationSDP(Bell_ABCD, verbose=0)
one_intermediate_latent_SDP.generate_relaxation([[],
                                                 [0],
                                                 [1],
                                                 [2],
                                                 [3],
                                                 [0, 1],
                                                 [0, 2],
                                                 [0, 3],
                                                 # [0, 2, 3]
                                                 ])
# one_intermediate_latent_SDP.generate_relaxation("npa2")
#
# one_intermediate_latent_SDP = InflationLP(Bell_ABCD, verbose=0)

one_intermediate_latent_SDP.set_objective(monogamy_objective, direction="max")
#
# one_intermediate_latent_SDP.set_objective({
#     "P[A_0=0]": -1,
#     "P[C_0=0]": -1,
#     "P[A_0=0 C_0=0]": 2,
# }, direction="max")

# one_intermediate_latent_SDP.set_objective({
#     "P[A_0=0]": -1,
#     "P[B_0=0]": -1,
#     "P[A_0=0 B_0=0]": 1,
#     "P[A_0=0 B_1=0]": 1,
#     "P[A_1=0 B_0=0]": 1,
#     "P[A_1=0 B_1=0]": -1,
# }, direction="max")

# one_intermediate_latent_SDP.set_distribution(P_CharlieGuessesAlice)
one_intermediate_latent_SDP.solve(solve_dual=False)
from scipy.sparse import coo_array
print(one_intermediate_latent_SDP.solution_object['status'])
print("Max objective", one_intermediate_latent_SDP.primal_objective)

sol_dict = one_intermediate_latent_SDP.solution_object['x']
ch_value = 0.0
for k, v in CH_objective.items():
    ch_value += sol_dict[k] * v
print("CH value:",ch_value)

# Z=one_intermediate_latent_SDP.solution_object['Z']
# rounded_Z = np.asarray(np.round(4*Z,3), dtype=int)
# print(rounded_Z)
# print(np.linalg.eigvals(Z))
# print(np.linalg.eigvals(rounded_Z))
# # print(one_intermediate_latent_SDP.certificate_as_string())
#
# for k,v in one_intermediate_latent_SDP.maskmatrices.items():
#     if k.name in monogamy_objective or k.name == "constant_term":
#         print(f"{k} ({monogamy_objective[k.name]})")
#         # print(np.asarray(v.todense(),dtype=int))
#         print(np.trace(np.matmul(
#             Z,
#             v.todense())))
#         print(np.trace(np.matmul(
#             rounded_Z,
#             v.todense())))