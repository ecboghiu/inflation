from inflation import InflationProblem, InflationLP
import numpy as np

Evans = InflationProblem({"U_AB": ["A", "B"],
                          "U_BC": ["B", "C"],
                          "B": ["A", "C"]},
                         outcomes_per_party=(2, 2, 2),
                         settings_per_party=(1, 1, 1),
                         inflation_level_per_source=(2, 2),  # TO BE MODIFIED
                         order=("A", "B", "C"))

p_A = [0.5, 0.5]
p_BC_cond_A = np.zeros((2, 2, 2))
p_BC_cond_A[0, 0, 0] = 0.3
p_BC_cond_A[1, 0, 0] = 0.7
p_BC_cond_A[0, 0, 1] = 0.7
p_BC_cond_A[1, 1, 1] = 0.3
p_BC_cond_A = 0.9 * p_BC_cond_A + 0.1 * (1 / 4)
p_ABC = np.zeros((2, 2, 2, 1, 1, 1))
for (b, c, a), v in np.ndenumerate(p_BC_cond_A):
    p_ABC[a, b, c] = v*p_A[a]

# print("Now considering NONFANOUT inflation (a.k.a. `SWIGs\INTERRUPTION`)")
#Evans_Interrupted = InflationLP(Evans,
#                                nonfanout=True,
#                                  verbose=1)
# Evans_Interrupted.set_distribution(p_ABC)
# Evans_Interrupted_bounds = dict()

# Evans_Interrupted.set_objective(objective={'<C_0_1_0_1>': 1},
#                                  direction='min')
# Evans_Interrupted.solve()
# Evans_Interrupted_bounds["P(C=1 do B##=0) >= "] = Evans_Interrupted.objective_value
# print("Objective value: ", Evans_Interrupted.objective_value)

#Evans_Interrupted.set_objective(objective={'<C_0_1_0_0>': 1},
#                                  direction='min')
# Evans_Interrupted.solve(verbose=0, dualise=False)
#Evans_Interrupted_bounds["P(C=1 do B##=0) <= "] = 1 - Evans_Interrupted.objective_value
#print("Objective value: ", 1 - Evans_Interrupted.objective_value)
#
# Evans_Interrupted.set_objective(objective={'<C_0_1_1_1>': 1},
#                                  direction='min')
#
# Evans_Interrupted.solve()
# Evans_Interrupted_bounds["P(C=1 do B##=1) >= "] = Evans_Interrupted.objective_value
#
# Evans_Interrupted.set_objective(objective={'<C_0_1_1_1>': 1},
#                                  direction='max')
# Evans_Interrupted.solve()
# Evans_Interrupted_bounds["P(C=1 do B##=1) <= "] = Evans_Interrupted.objective_value
# for k, v in Evans_Interrupted_bounds.items():
#     print(f"{k}{v:.3f}")



print("Now considering FANOUT inflation (a.k.a. `UNPACKING`)")
Evans_Unpacked = InflationLP(Evans,
                             nonfanout=False,
                             verbose=1)
Evans_Unpacked.set_distribution(p_ABC)
Evans_Unpacked_bounds = dict()
#
# Evans_Unpacked.set_objective(objective={'<C_0_1_0_1>': 1},
#                                  direction='min')
#
# # for k, v in Evans_Unpacked._prepare_solver_arguments().items():
# #     print(f"{k}: {v}")
#
# Evans_Unpacked.solve()
# Evans_Unpacked_bounds["P(C=1 do B##=0) >= "] = Evans_Unpacked.objective_value
#
Evans_Unpacked.set_objective(objective={'<C_0_1_0_0>': 1},
                             direction='min')
Evans_Unpacked.solve(verbose=0)
print("Status: ", Evans_Unpacked.objective_value)
Evans_Unpacked_bounds["P(C=1 do B##=0) <= "] = 1 - Evans_Unpacked.objective_value
print("Objective value: ", 1 - Evans_Unpacked.objective_value)
#
# Evans_Unpacked.set_objective(objective={'<C_0_1_1_1>': 1},
#                                  direction='min')
#
# Evans_Unpacked.solve()
# Evans_Unpacked_bounds["P(C=1 do B##=1) >= "] = Evans_Unpacked.objective_value
#
# Evans_Unpacked.set_objective(objective={'<C_0_1_1_1>': 1},
#                                  direction='max')
# Evans_Unpacked.solve()
# Evans_Unpacked_bounds["P(C=1 do B##=1) <= "] = Evans_Unpacked.objective_value
# for k, v in Evans_Unpacked_bounds.items():
#     print(f"{k}{v:.3f}")
