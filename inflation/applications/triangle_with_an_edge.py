from inflation import InflationProblem, InflationLP
import numpy as np

P_GHZ = np.zeros((2, 2, 2, 1, 1, 1))
P_GHZ[0, 0, 0] = 1 / 2
P_GHZ[1, 1, 1] = 1 / 2

GHZ_objective = {'P[B=0|do(A=0)]': 1, 'P[B=0|do(A=1)]': -1}

triangle_plus_edge_nonfanout = InflationProblem({"lambda": ["A", "B"],
                                                 "mu": ["B", "C"],
                                                 "sigma": ["A", "C"],
                                                 "A": ["B"]},
                                                classical_sources=None,
                                                outcomes_per_party=[2, 2, 2],
                                                settings_per_party=[1, 1, 1],
                                                inflation_level_per_source=[2,
                                                                            2,
                                                                            2],
                                                order=['A', 'B', 'C'],
                                                verbose=1)
triangle_web_nonfanout_LP = InflationLP(triangle_plus_edge_nonfanout)
for m in triangle_web_nonfanout_LP.knowable_atoms:
    print(m)
for m in triangle_web_nonfanout_LP.do_conditional_atoms:
    print(m)
parameters_we_care_about = triangle_web_nonfanout_LP.knowable_atoms + triangle_web_nonfanout_LP.do_conditional_atoms

triangle_web_nonfanout_LP.set_distribution(P_GHZ, use_lpi_constraints=True)
triangle_web_nonfanout_LP.set_objective(
    objective={'P[B=0|do(A=0)]': 1, 'P[B=0|do(A=1)]': -1},
    direction='min')
triangle_web_nonfanout_LP.solve(solve_dual=False, verbose=0)
print(f"Objective value: {triangle_web_nonfanout_LP.objective_value:.3f}")
for m in parameters_we_care_about:
    v = triangle_web_nonfanout_LP.solution_object['x'][m.name]
    print(f"{m}->{v}")
