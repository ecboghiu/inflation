import numpy as np
from inflation import InflationProblem, InflationSDP

tripartite_Bell = InflationProblem(
    dag={"rho_ABC": ["A", "B", "C"]},
    outcomes_per_party=[2, 2, 2],
    settings_per_party=[2, 2, 4],
    inflation_level_per_source=[1],
    order=['A', 'B', 'C'],
    verbose=2)

print(tripartite_Bell._lexrepr_to_names)

print("Before manually adjusting noncommutation relations:")
print(tripartite_Bell._default_notcomm.astype(int))
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
print(tripartite_Bell._default_notcomm.astype(int))

# Let the distribution be such that Charlie guesses Alice's outcome with perfect certainty
v = 1 / np.sqrt(2)
P_CharlieGuessesAlice = np.full((2,2,2,2,2,4), fill_value=np.nan)
for a,b,c,x,y in np.ndindex(2,2,2,2,2):
    z = 2*x + y
    if a != c:
        P_CharlieGuessesAlice[a, b, c, x, y, z] = 0
    else:
        P_CharlieGuessesAlice[a, b, c, x, y, z] = (1 / 4) * ((1 + (v) * ((-1) ** (a + b + x * y))))


one_intermediate_latent_SDP = InflationSDP(tripartite_Bell, verbose=1)
one_intermediate_latent_SDP.generate_relaxation("npa2")

# w = np.sqrt(2)/2
w = 1
one_intermediate_latent_SDP.set_objective({
    "P[A_0=0]": -1-w,
    "P[B_0=0]": -1,
    "P[A_0=0 B_0=0]": 1,
    "P[A_0=0 B_1=0]": 1,
    "P[A_1=0 B_0=0]": 1,
    "P[A_1=0 B_1=0]": -1,
    "P[C_0=0]": -w,
    "P[A_0=0 C_0=0]": w,
}, direction="max")
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
one_intermediate_latent_SDP.solve()
print(one_intermediate_latent_SDP.solution_object['status'])
print(one_intermediate_latent_SDP.primal_objective)
# print(one_intermediate_latent_SDP.certificate_as_string())