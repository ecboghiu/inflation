from inflation import InflationProblem, InflationLP
import numpy as np
#
# P_W = np.zeros((2, 2, 2, 1, 1, 1))
# P_W[1, 0, 0] = 1 / 3
# P_W[0, 1, 0] = 1 / 3
# P_W[0, 0, 1] = 1 / 3

P_GHZ = np.zeros((2, 2, 2, 1, 1, 1))
P_GHZ[0, 0, 0] = 1 / 2
P_GHZ[1, 1, 1] = 1 / 2

GHZ_objective = {'P[B=0|do(A=0)]': 1, 'P[B=0|do(A=1)]': -1}

# triangle_plus_edge_fanout = InflationProblem({"lambda": ["A", "B"],
#                                              "mu": ["B", "C"],
#                                              "sigma": ["A", "C"],
#                                               "A": ["B"]},
#                                             classical_sources=['lambda', 'mu', 'sigma'],
#                                             outcomes_per_party=[2, 2, 2],
#                                             settings_per_party=[1, 1, 1],
#                                             inflation_level_per_source=[1, 1, 5],
#                                             order=['A', 'B', 'C'],
#                                             verbose=1)
# triangle_web_with_fanout_LP = InflationLP(triangle_plus_edge_fanout)
#
# triangle_web_with_fanout_LP.set_distribution(P_GHZ, use_lpi_constraints=True)
# triangle_web_with_fanout_LP.set_objective(objective={'P[B=0|do(A=0)]': 1, 'P[B=0|do(A=1)]': -1},
#                              direction='min')
# triangle_web_with_fanout_LP.solve(solve_dual=False, verbose=1)
# print(f"Objective value: {triangle_web_with_fanout_LP.objective_value:.3f}")
# for k in GHZ_objective.keys():
#     print(f"{k} --> {triangle_web_with_fanout_LP.solution_object['x'][k]}")

triangle_plus_edge_nonfanout = InflationProblem({"lambda": ["A", "B"],
                                             "mu": ["B", "C"],
                                             "sigma": ["A", "C"],
                                              "A": ["B"]},
                                            classical_sources=None,
                                            outcomes_per_party=[2, 2, 2],
                                            settings_per_party=[1, 1, 1],
                                            inflation_level_per_source=[2, 2, 2],
                                            order=['A', 'B', 'C'],
                                            verbose=1)
triangle_web_nonfanout_LP = InflationLP(triangle_plus_edge_nonfanout)

triangle_web_nonfanout_LP.set_distribution(P_GHZ, use_lpi_constraints=True)
triangle_web_nonfanout_LP.set_objective(objective={'P[B=0|do(A=0)]': 1, 'P[B=0|do(A=1)]': -1},
                             direction='min')
triangle_web_nonfanout_LP.solve(solve_dual=False, verbose=0)
print(f"Objective value: {triangle_web_nonfanout_LP.objective_value:.3f}")
# for k in GHZ_objective.keys():
#     print(f"{k} --> {triangle_web_nonfanout_LP.solution_object['x'][k]}")

# print(triangle_web_with_fanout_LP.certificate_as_string(clean=True))

# print("Atomic Knowables:")
# for m in triangle_web_with_fanout_LP.monomials:
#     if m.is_knowable and m.is_atomic:
#         print(f"{m.name} --> {triangle_web_with_fanout_LP.solution_object['x'][m.name]}")
# print("Atomic Do Conditionals:")
# for m in triangle_web_with_fanout_LP.monomials:
#     if m.is_do_conditional and not m.is_knowable and m.is_atomic:
#         print(f"{m.name} --> {triangle_web_with_fanout_LP.solution_object['x'][m.name]}")

# triangle_web_with_fanout_LP.set_objective(objective={'P[B=0|do(A=1)]': 1},
#                              direction='max')
# triangle_web_with_fanout_LP.solve(solve_dual=False, verbose=1)
# print(f"Objective value: {triangle_web_with_fanout_LP.objective_value:.3f}")
# print(triangle_web_with_fanout_LP.certificate_as_string(clean=True))