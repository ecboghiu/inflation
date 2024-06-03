import numpy as np
from inflation import InflationProblem, InflationSDP



tripartite_Bell = InflationProblem(
    dag={"rho_ABC": ["A", "B", "C"]},
    # dag={"rho_ABC": ["A", "L"], "L": ["B", "C"]},
    outcomes_per_party=[2, 2, 2],
    settings_per_party=[2, 2, 4],
    inflation_level_per_source=[1],
    # nonclassical_intermediate_latents=["L"],
    order=['A', 'B', 'C'],
    verbose=2)

# We want to impose that if B and C have different settings they do not commute.

print(tripartite_Bell._lexorder)

print("Before manually adjusting noncommutation relations:")
print(tripartite_Bell._default_notcomm.astype(int))
n_ops = tripartite_Bell._nr_operators
# for i, j in np.ndindex((n_ops,n_ops)):
#     op_i = tripartite_Bell._lexorder[i]
#     op_j = tripartite_Bell._lexorder[j]
#     if (op_i[0] == 2) and (op_j[0] == 3):
#         y_B = op_i[-2]
#         x_C, y_C = np.divmod(op_j[-2], 2)
#         if y_B != y_C:
#             tripartite_Bell._default_notcomm[i, j] = True
#             tripartite_Bell._default_notcomm[j, i] = True
for i, j in np.ndindex((n_ops,n_ops)):
    op_i = tripartite_Bell._lexorder[i]
    op_j = tripartite_Bell._lexorder[j]
    if (op_i[0] == 1) and (op_j[0] == 3):
        x_A = op_i[-2]
        x_C, y_C = np.divmod(op_j[-2], 2)
        if x_A != x_C:
            tripartite_Bell._default_notcomm[i, j] = True
            tripartite_Bell._default_notcomm[j, i] = True
print("After manually adjusting noncommutation relations:")
print(tripartite_Bell._default_notcomm.astype(int))

v = 1 / np.sqrt(2)
P_Dani = np.full((2,2,2,2,2,4), fill_value=np.nan)
for a,b,c,x,y in np.ndindex(2,2,2,2,2):
    z = 2*x + y
    if b != c:
        P_Dani[a, b, c, x, y, z] = 0
    else:
        P_Dani[a, b, c, x, y, z] = (1 / 4) * ((1 + (v) * ((-1) ** (a + b + x * y))))
for x, y in np.ndindex((2,2)):
    z = 2 * x + y
    print(P_Dani[:, :, :, x, y, z])

one_intermediate_latent_SDP = InflationSDP(tripartite_Bell, verbose=2, include_all_outcomes=True)
one_intermediate_latent_SDP.generate_relaxation("npa2")

one_intermediate_latent_SDP.set_distribution(P_Dani)
print(one_intermediate_latent_SDP.known_moments)
one_intermediate_latent_SDP.solve()
print(one_intermediate_latent_SDP.solution_object['status'])