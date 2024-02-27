"""
A-->B-->C-->D
Hidden = {X,Y,Z}
X->{BCD}
Y-{AC}
Z->{AD}
"""

from inflation import InflationProblem, InflationLP
import numpy as np

p_Bonet_violating_conditional = np.zeros((3,2,2,3,1,1,1,1))
for (a,b,c) in np.ndindex((3,2,2)):
    if a < 2:
        p_Bonet_violating_conditional[a,b,c,a] = 0.5*int(np.logical_and(a,b) == np.logical_xor(b,c))
    if a==2:
        p_Bonet_violating_conditional[a, b, c, a] = 0.5 * int(c == 1)

p_Bonet_violating = np.array([10/21,10/21,1/21]).reshape((3,1,1,1,1,1,1,1))*p_Bonet_violating_conditional
for a in range(3):
    print(p_Bonet_violating[a].ravel())

HardDAG = InflationProblem({"X": ["B", "C", "D"],
                            "Y": ["A", "C"],
                            "Z": ["A", "D"],
                            "A": ["B"],
                            "B": ["C"],
                            "C": ["D"]},
                           outcomes_per_party=(3, 2, 2, 3),
                           settings_per_party=(1, 1, 1, 1),
                           inflation_level_per_source=(1, 2, 2),
                           order=("A", "B", "C", "D"),
                           classical_sources="all")

HardDAG_Unpacked = InflationLP(HardDAG,
                             nonfanout=False,
                             verbose=2,
                             local_level=1)

semiknown_usage = False
HardDAG_Unpacked.set_distribution(p_Bonet_violating, use_lpi_constraints=semiknown_usage)

print(f"Known Values: \n")
for k, v in HardDAG_Unpacked.known_moments.items():
    print(f"{k}: {v}")
HardDAG_Unpacked.solve(solve_dual=False, verbose=2)
print("Status: ", HardDAG_Unpacked.status)
print(HardDAG_Unpacked.certificate_as_probs())