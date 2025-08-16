# import sys
# sys.path.append('/Users/pedrolauand/inflation')
from inflation import InflationProblem, InflationLP, InflationSDP
import numpy as np

def ring_problem(inflation_level: int, nof_outcomes: int = 2) -> InflationProblem:
    return InflationProblem(
        dag={"i1": ["A"],
             "i2": ["A"], },
        outcomes_per_party=(nof_outcomes,),
        settings_per_party=(1,),
        classical_sources=None,
        inflation_level_per_source=(inflation_level,inflation_level),
        order=["A"],
        really_just_one_source=True) # NEW!!


prob = ring_problem(4, 2)
# prob.add_symmetries(prob._setting_specific_outcome_relabelling_symmetries)
ring_SDP = InflationSDP(prob, verbose=2, include_all_outcomes=False)
ring_SDP.generate_relaxation("physical2")
# ring_SDP = InflationLP(prob, verbose=2)


print("Quantum inflation **nonfanout/commuting** factors:")
print(ring_SDP.physical_atoms)



def tetrahedron_problem(inflation_level: int, nof_outcomes: int = 2) -> InflationProblem:
    return InflationProblem(
        dag={"i1": ["A"],
             "i2": ["A"],
             "i3": ["A"]},
        outcomes_per_party=(nof_outcomes,),
        settings_per_party=(1,),
        classical_sources=None,
        inflation_level_per_source=[inflation_level]*3,
        order=["A"],
        really_just_one_source=True)

tet_prob = tetrahedron_problem(5, 2)
print(tet_prob._lexrepr_to_names[::2])
tet_LP = InflationLP(tet_prob, verbose=2, include_all_outcomes=False)

print("Nonfanout inflation atomic factors:")
print(tet_LP.atomic_factors)