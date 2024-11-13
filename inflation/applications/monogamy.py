import numpy as np
from numpy import dtype

from inflation import InflationProblem, InflationSDP

# Although C has 4 settings, we only refer to one in the objective, so we'll lie WLOG in the input.
tripartite_Bell = InflationProblem(
    dag={"rho_ABC": ["A", "B", "C"]},
    outcomes_per_party=[2, 2, 2],
    settings_per_party=[2, 2, 1],
    inflation_level_per_source=[1],
    order=['A', 'B', 'C'],
    verbose=0)

print(list(map(str, tripartite_Bell._lexrepr_to_names[::2])))

print("Before manually adjusting noncommutation relations:")
print(tripartite_Bell._default_notcomm.astype(int)[::2,::2])
n_ops = tripartite_Bell._nr_operators
# If we want to impose that when B and C have different settings they do not commute:
for i, j in np.ndindex((n_ops,n_ops)):
    op_i = tripartite_Bell._lexorder[i]
    op_j = tripartite_Bell._lexorder[j]
    if (op_i[0] == 2) and (op_j[0] == 3):
        y_B = op_i[-2]
        x_C, y_C = np.divmod(op_j[-2], 2)
        if y_B != y_C:
            tripartite_Bell._default_notcomm[i, j] = True
            tripartite_Bell._default_notcomm[j, i] = True
# If we want to impose that when A and C have different settings they do not commute:
# for i, j in np.ndindex((n_ops,n_ops)):
#     op_i = tripartite_Bell._lexorder[i]
#     op_j = tripartite_Bell._lexorder[j]
#     if (op_i[0] == 1) and (op_j[0] == 3):
#         x_A = op_i[-2]
#         x_C, y_C = np.divmod(op_j[-2], 2)
#         if x_A != x_C:
#             tripartite_Bell._default_notcomm[i, j] = True
#             tripartite_Bell._default_notcomm[j, i] = True
print("After manually adjusting noncommutation relations:")
print(tripartite_Bell._default_notcomm.astype(int)[::2,::2])

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
monogamy_objective = {
    "P[A_0=0]": -1-w,
    "P[B_0=0]": -1,
    "P[A_0=0 B_0=0]": 1,
    "P[A_0=0 B_1=0]": 1,
    "P[A_1=0 B_0=0]": 1,
    "P[A_1=0 B_1=0]": -1,
    "P[C=0]": -w,
    "P[A_0=0 C=0]": 2*w}
CH_objective = {
    "P[A_0=0]": -1,
    "P[B_0=0]": -1,
    "P[A_0=0 B_0=0]": 1,
    "P[A_0=0 B_1=0]": 1,
    "P[A_1=0 B_0=0]": 1,
    "P[A_1=0 B_1=0]": -1}

one_intermediate_latent_SDP = InflationSDP(tripartite_Bell, verbose=0)
one_intermediate_latent_SDP.generate_relaxation([[],
                                                 [0],
                                                 [1],
                                                 [2],
                                                 [0, 1],
                                                 [0, 2]])
# one_intermediate_latent_SDP.generate_relaxation("npa2")

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