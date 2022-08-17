import numpy as np

from causalinflation import InflationProblem, InflationSDP

# Let's study the UC-DAG
UC_SDP = InflationSDP(InflationProblem(dag={"U_AB": ["A", "B"],
                           "U_BC": ["B", "C"],
                           "B": ["A", "C"]},
                      names=('A', 'B', 'C'),
                      outcomes_per_party=[2, 2, 2],
                      settings_per_party=[1, 1, 1],
                      inflation_level_per_source=[1, 2]))

# Example support for UC
support1_UC = np.zeros((2, 2, 2, 1, 1, 1), dtype=bool)
for (a, b, c, x, y, z) in np.ndindex(2, 2, 2, 1, 1, 1):
    support1_UC[a, b, c, x, y, z] = (a == c and b == 0)

# UC_SDP.generate_relaxation('local1')  # Other good choices are 'npa2' or 'local1' etc...
# print("SDP Relaxation has been generated, now setting distribution.")
# UC_SDP.set_distribution(support1_UC, treat_as_support=True)
# print("Distribution (or, rather, support) has been set, now initiating optimizer.")
# UC_SDP.solve()
# print(UC_SDP.status)

# Let's study the IV-DAG
IV_SDP = InflationSDP(InflationProblem(dag={"U_AB": ["A", "B"],
                           "A": ["B"]},
                      names=('A', 'B'),
                      outcomes_per_party=[2, 2],
                      settings_per_party=[2, 1],
                      inflation_level_per_source=[1]))
# Example support for IV
support1_IV = np.zeros((2, 2, 2, 1), dtype=bool)
for (a, b, x, _) in np.ndindex(2, 2, 2, 1):
    support1_UC[a, b, x, 0] = (b == x and a == 0)

IV_SDP.generate_relaxation('npa3')  # Other good choices are 'npa2' or 'local1' etc...
print("SDP Relaxation has been generated, now setting distribution.")
IV_SDP.set_distribution(support1_IV, treat_as_support=True)
print("Distribution (or, rather, support) has been set, now initiating optimizer.")
IV_SDP.solve()
print(IV_SDP.status)





