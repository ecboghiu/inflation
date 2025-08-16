from inflation import InflationProblem, InflationSDP
import numpy as np


def ring_problem(inflation_level: int, nof_outcomes: int = 2) -> InflationProblem:
    inf_prob = InflationProblem(
        dag={"i1": ["A"],
             "i2": ["A"], },
        outcomes_per_party=(nof_outcomes,),
        settings_per_party=(1,),
        classical_sources=None,
        inflation_level_per_source=(inflation_level,inflation_level),
        order=["A"],
        really_just_one_source=True)
    return inf_prob


prob = ring_problem(4, 2)
prob.add_symmetries(prob._setting_specific_outcome_relabelling_symmetries)

from postquantum_2outcomes import prob_line, prob_loop

ring_SDP = InflationSDP(prob, verbose=2, include_all_outcomes=False)
ring_SDP.generate_relaxation("physical2")

print("Quantum inflation **nonfanout/commuting** factors:")
print(ring_SDP.physical_atoms)

values={
    'P[A^{1,1}=0]': prob_loop(1),
    'P[A^{1,2}=0]': prob_line(1),
    'P[A^{1,2}=0 A^{2,1}=0]': prob_loop(2),
    'P[A^{1,2}=0 A^{2,3}=0]': prob_line(2),
    'P[A^{1,2}=0 A^{2,3}=0 A^{3,1}=0]': prob_loop(3),
    'P[A^{1,2}=0 A^{2,3}=0 A^{3,4}=0]': prob_line(3),
    'P[A^{1,2}=0 A^{2,3}=0 A^{3,4}=0 A^{4,1}=0]': prob_loop(4)
}
ring_SDP.update_values(values=values, only_specified_values=False)
print(ring_SDP.known_moments)

ring_SDP.solve(solve_dual=False)
print(ring_SDP.status)

