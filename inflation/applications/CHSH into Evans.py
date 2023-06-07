from inflation import InflationProblem, InflationLP
import numpy as np

p_Q = np.zeros((2, 2, 2, 1, 1, 1), dtype=float)
q = [1 / 2, 1 / 2]
p_Q_do = np.zeros((2, 2, 2), dtype=float)
v = 1 / np.sqrt(2)

for a in range(2):
    for b in range(2):
        for c in range(2):
            p_Q_do[a, b, c] = q[c] * (1 / 2)
            p_Q[a, b, c] = (q[c]) * (1 / 4) * (
            (1 + (v) * ((-1) ** (a + b + b * c))))

Evans = InflationProblem({"U_AB": ["A", "B"],
                          "U_BC": ["B", "C"],
                          "B": ["A", "C"]},
                         outcomes_per_party=(2, 2, 2),
                         settings_per_party=(1, 1, 1),
                         inflation_level_per_source=(2, 2),  # TO BE MODIFIED
                         order=("A", "B", "C"))

Evans_Unpacked = InflationLP(Evans,
                             nonfanout=False,
                             verbose=2)
Evans_Unpacked.set_distribution(p_Q)
known_do_values = {f'<C_0_1_{b}_{c}>': q[c]
                   for c in range(1)
                   for b in range(2)}
# known_do_values.update({f'<C_0_2_{b}_{c}>': q[c]
#                         for c in range(1)
#                         for b in range(2)})
# known_do_values.update({f'<A_1_0_{b}_{a}>': 1 / 2
#                         for a in range(1)
#                         for b in range(2)})
known_do_values.update({f'<A_2_0_{b}_{a}>': 1 / 2
                        for a in range(1)
                        for b in range(2)})
# known_do_values.update({f'<A_1_0_{b1}_{a}>*<C_0_1_{b2}_{c}>': q[c]*1 / 2
#                         for a in range(2)
#                         for c in range(2)
#                         for b1 in range(2)
#                         for b2 in range(2)})
Evans_Unpacked.set_values(known_do_values,
                          use_lpi_constraints=False)
print(f"Known Values: {Evans_Unpacked.known_moments}")
Evans_Unpacked.solve(dualise=False, verbose=2)
print(Evans_Unpacked.status)
x_dict = {n: np.round(v, decimals=3) for n, v in Evans_Unpacked.solution_object['x'].items() if np.abs(v)>1e-10}
print(x_dict)
print(Evans_Unpacked.certificate_as_probs())


