import numpy as np


def PR_box_with_visibility(a: int, b: int, x: int, y: int, v=np.sqrt(2)/2):
    return (1 + v * (-1) ** (a + b + x * y)) / 4

def NSBoxType1(v=np.sqrt(2)/2):
    tripartite_box = np.zeros((2, 2, 2, 2, 1, 2), dtype=object)
    for (x, z, a, b, c) in np.ndindex((2, 2, 2, 2, 2)):
        tripartite_box[a, b, c, x, 0, z] = 1/2 * PR_box_with_visibility(a, np.mod(b+c,2), x, c, v)
    return tripartite_box


def NSBoxType2(v=np.sqrt(2)/2, post_selection_probability=1 / 4):
    tripartite_box = np.zeros((2, 2, 2, 2, 1, 2), dtype=object)
    for (a, c, x, z) in np.ndindex((2, 2, 2, 2)):
        tripartite_box[
            a, 0, c, x, 0, z] = post_selection_probability * PR_box_with_visibility(
            a, c, x, z, v)
    tripartite_box[:, 1] = 1/4 - tripartite_box[:, 0]
    return tripartite_box
def NSBoxHalfAndHalf(v=np.sqrt(2)/2, post_selection_probability=1/4):
    return NSBoxType1(v) / 2 + NSBoxType2(v, post_selection_probability=post_selection_probability) / 2

def NSBoxFrom1to2(w: float=1, v=np.sqrt(2)/2, post_selection_probability=1/4):
    return (1-w)*NSBoxType1(v) + w*NSBoxType2(v, post_selection_probability=post_selection_probability)
def NSBoxFrom2to1(w: float=1, v=np.sqrt(2)/2, post_selection_probability=1/4):
    return (1-w)*NSBoxType2(v, post_selection_probability=post_selection_probability) + w*NSBoxType1(v)

def NSBoxFromNoisy1to2(w: float=1, v=np.sqrt(2)/2, post_selection_probability=1/4):
    return (1-w)*NSBoxType1(v=0) + w*NSBoxType2(v, post_selection_probability=post_selection_probability)
def NSBoxFromNoisy2to1(w: float=1, v=np.sqrt(2)/2, post_selection_probability=1/4):
    return (1-w)*NSBoxType2(v=0, post_selection_probability=post_selection_probability) + w*NSBoxType1(v)

from inflation import InflationProblem, InflationSDP, InflationLP

simplest_bilocal_scenario_no_inflation = InflationProblem({
    "lambda": ["a", "b"],
    "mu": ["b", "c"]},
    outcomes_per_party=[2, 2, 2],
    settings_per_party=[2, 1, 2],
    inflation_level_per_source=[1, 1],
    order=['a', 'b', 'c'],
    verbose=1)

# quantum_bilocal_SDP = InflationSDP(simplest_bilocal_scenario)
# quantum_bilocal_SDP.generate_relaxation("physical")


from sympy import Symbol
from inflation.sdp.optimization_utils import max_within_feasible

simplest_bilocal_lp = InflationLP(simplest_bilocal_scenario_no_inflation,
                                  nonfanout=False,
                                  verbose=1)

simplest_bilocal_lp.set_distribution(
    NSBoxFrom1to2(w=1/2, v=np.sqrt(2)/2, post_selection_probability=1/4))
simplest_bilocal_lp.solve(interpreter="solve_Gurobi")
#
# #
# # print("TESTING CHARLIES-OUTCOME-AS-BOB'S SETTING BOX")
# # simplest_bilocal_lp.set_distribution(
# #     NSBoxType1(v=Symbol("v")))
# # symbolic_moments = simplest_bilocal_lp.known_moments.copy()
# # v, cert = max_within_feasible(simplest_bilocal_lp,
# #                               symbolic_moments,
# #                               method="dual",
# #                               return_last_certificate=True,
# #                               verbose=1,
# #                               solve_dual=True,
# #                               use_lpi_constraints=False)
# # print(f"Critical visibility via dual cert: {v}: via \n{cert}")
# #
# # print("TESTING POST-SELECTION BOX")
# # simplest_bilocal_lp.set_distribution(
# #     NSBoxType2(v=Symbol("v"), post_selection_probability=1 / 2))
# # symbolic_moments = simplest_bilocal_lp.known_moments.copy()
# # v, cert = max_within_feasible(simplest_bilocal_lp,
# #                               symbolic_moments,
# #                               method="dual",
# #                               return_last_certificate=True,
# #                               verbose=1,
# #                               solve_dual=True,
# #                               use_lpi_constraints=False)
# # print(f"Critical visibility via dual cert: {v}: via \n{cert}")
#
# print("TESTING NOISY-PR mixture nonclassicality, post_selection_probability = 1/2")
# simplest_bilocal_lp.set_distribution(
#     NSBoxHalfAndHalf(v=Symbol("v"), post_selection_probability=1 / 2))
# symbolic_moments = simplest_bilocal_lp.known_moments.copy()
# v, cert = max_within_feasible(simplest_bilocal_lp,
#                               symbolic_moments,
#                               method="dual",
#                               return_last_certificate=True,
#                               verbose=1,
#                               solve_dual=True,
#                               use_lpi_constraints=False)
# print(f"Critical visibility via dual cert: {v}: via \n{cert}")
#
simplest_bilocal_scenario_with_inflation = InflationProblem({
    "lambda": ["a", "b"],
    "mu": ["b", "c"]},
    outcomes_per_party=[2, 2, 2],
    settings_per_party=[2, 1, 2],
    inflation_level_per_source=[1, 2],
    order=['a', 'b', 'c'],
    verbose=0)
simplest_bilocal_scenario_with_inflation_lp = InflationLP(
    simplest_bilocal_scenario_with_inflation,
    nonfanout=False,
    verbose=0)











print("\n")
print("TESTING NOISY-PR equal mixture nonclassicality, post_selection_probability = 1/4")
simplest_bilocal_scenario_with_inflation_lp.set_distribution(
    NSBoxHalfAndHalf(Symbol("v"), post_selection_probability=1/4))
symbolic_moments = simplest_bilocal_scenario_with_inflation_lp.known_moments.copy()
v, cert = max_within_feasible(simplest_bilocal_scenario_with_inflation_lp,
                              symbolic_moments,
                              method="dual",
                              return_last_certificate=True,
                              verbose=1,
                              solve_dual=True,
                              use_lpi_constraints=False,
                              # solverparameters={mosek.iparam.optimizer: mosek.optimizertype.dual_simplex}
                              )
print(f"Critical visibility via dual cert: {v}: via \n{cert}")
#
# print("\n")
# print("TESTING NOISY-PR mixture nonquantumness, post_selection_probability = 1/4")
# quantum_bilocal_SDP.set_distribution(
#     NSBoxHalfAndHalf(Symbol("v"), post_selection_probability=1 / 4))
# symbolic_moments = quantum_bilocal_SDP.known_moments.copy()
# v, cert = max_within_feasible(quantum_bilocal_SDP,
#                               symbolic_moments,
#                               method="dual",
#                               return_last_certificate=True,
#                               verbose=1,
#                               solve_dual=True,
#                               use_lpi_constraints=False,
#                               # solverparameters={mosek.iparam.optimizer: mosek.optimizertype.dual_simplex}
#                               )
# print(f"Critical visibility via dual cert: {v}: via \n{cert}")
print("\n")
print("TESTING noiseless Tsirelson Box mixtures other way  [CRITICAL FINDING]")
simplest_bilocal_scenario_with_inflation_lp.set_distribution(
    NSBoxFrom2to1(Symbol("w"), v=np.sqrt(2)/2, post_selection_probability=1/4))
symbolic_moments = simplest_bilocal_scenario_with_inflation_lp.known_moments.copy()
w, cert = max_within_feasible(simplest_bilocal_scenario_with_inflation_lp,
                              symbolic_moments,
                              method="dual",
                              return_last_certificate=True,
                              verbose=1,
                              solve_dual=True,
                              use_lpi_constraints=False
                              )
print(f"Critical weight of charlie-as-setting box: {w}: via \n{cert}")

print("\n")
print("TESTING noiseless Tsirelson Box mixtures")
simplest_bilocal_scenario_with_inflation_lp.set_distribution(
    NSBoxFrom1to2(Symbol("w"), v=np.sqrt(2)/2, post_selection_probability=1/4))
symbolic_moments = simplest_bilocal_scenario_with_inflation_lp.known_moments.copy()
w, cert = max_within_feasible(simplest_bilocal_scenario_with_inflation_lp,
                              symbolic_moments,
                              method="dual",
                              return_last_certificate=True,
                              verbose=1,
                              solve_dual=True,
                              use_lpi_constraints=False,
                              # solverparameters={mosek.iparam.optimizer: mosek.optimizertype.dual_simplex}
                              )
print(f"Critical weight of postselection box: {w}: via \n{cert}")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# print("\n")
# print("TESTING noise resistance of Charlie-as-setting Box")
# simplest_bilocal_scenario_with_inflation_lp.set_distribution(
#     NSBoxFromNoisy2to1(Symbol("w"), v=np.sqrt(2)/2))
# symbolic_moments = simplest_bilocal_scenario_with_inflation_lp.known_moments.copy()
# w, cert = max_within_feasible(simplest_bilocal_scenario_with_inflation_lp,
#                               symbolic_moments,
#                               method="dual",
#                               return_last_certificate=True,
#                               verbose=0,
#                               solve_dual=True,
#                               use_lpi_constraints=False
#                               )
# print(f"Noise threshold for charlie-as-setting box: {w}: via \n{cert}")
# print("\n")
# print("TESTING noise resistance of postselection Box")
# simplest_bilocal_scenario_with_inflation_lp.set_distribution(
#     NSBoxFromNoisy1to2(Symbol("w"), v=np.sqrt(2)/2, post_selection_probability=1/4))
# symbolic_moments = simplest_bilocal_scenario_with_inflation_lp.known_moments.copy()
# w, cert = max_within_feasible(simplest_bilocal_scenario_with_inflation_lp,
#                               symbolic_moments,
#                               method="dual",
#                               return_last_certificate=True,
#                               verbose=0,
#                               solve_dual=True,
#                               use_lpi_constraints=False
#                               )
# print(f"Noise threshold for postselection box: {w}: via \n{cert}")
#
#
#
# print("\n")
# print("TESTING NOISY-PR 70% weighted charlie-as-setting, 30% weighted postselection, post_selection_probability = 1/4")
# simplest_bilocal_scenario_with_inflation_lp.set_distribution(
#     NSBoxFrom2to1(w=0.7, v=Symbol("v"), post_selection_probability=1/4))
# symbolic_moments = simplest_bilocal_scenario_with_inflation_lp.known_moments.copy()
# v, cert = max_within_feasible(simplest_bilocal_scenario_with_inflation_lp,
#                               symbolic_moments,
#                               method="dual",
#                               return_last_certificate=True,
#                               verbose=0,
#                               solve_dual=True,
#                               use_lpi_constraints=False,
#                               # solverparameters={mosek.iparam.optimizer: mosek.optimizertype.dual_simplex}
#                               )
# print(f"Critical visibility via dual cert: {v}: via \n{cert}")
#
#
#
# print("\n")
# print("TESTING NOISY-PR 70% weighted charlie-as-setting, 30% weighted postselection, post_selection_probability = 1/5")
# simplest_bilocal_scenario_with_inflation_lp.set_distribution(
#     NSBoxFrom2to1(w=0.7, v=Symbol("v"), post_selection_probability=1/5))
# symbolic_moments = simplest_bilocal_scenario_with_inflation_lp.known_moments.copy()
# v, cert = max_within_feasible(simplest_bilocal_scenario_with_inflation_lp,
#                               symbolic_moments,
#                               method="dual",
#                               return_last_certificate=True,
#                               verbose=0,
#                               solve_dual=True,
#                               use_lpi_constraints=False,
#                               # solverparameters={mosek.iparam.optimizer: mosek.optimizertype.dual_simplex}
#                               )
# print(f"Critical visibility via dual cert: {v}: via \n{cert}")
#
# # print("\n")
# # print("TESTING noiseless Tsirelson Box mixtures other way  [Lower postselection probability]")
# # simplest_bilocal_scenario_with_inflation_lp.set_distribution(
# #     NSBoxFrom2to1(Symbol("w"), v=np.sqrt(2)/2, post_selection_probability=1/100))
# # symbolic_moments = simplest_bilocal_scenario_with_inflation_lp.known_moments.copy()
# # w, cert = max_within_feasible(simplest_bilocal_scenario_with_inflation_lp,
# #                               symbolic_moments,
# #                               method="dual",
# #                               return_last_certificate=True,
# #                               verbose=0,
# #                               solve_dual=True,
# #                               use_lpi_constraints=False
# #                               )
# # print(f"Critical weight of charlie-as-setting box: {w}: via \n{cert}")

simplest_bilocal_scenario_with_inflation_lp.set_distribution(
    NSBoxFrom2to1(Symbol("w"), v=Symbol("v"), post_selection_probability=Symbol("q")))
symbolic_moments = {str(k): v for k, v in simplest_bilocal_scenario_with_inflation_lp.known_moments.items()}
dual_cert = {'pa(0|1)*pc(0|0)': 0.022727272727272707, 'pbc(00|00)': 0.022727272727272707, 'pc(0|0)*pab(00|00)': 0.022727272727272794, 'pc(0|0)*pab(00|10)': -0.022727272727272707, 'pabc(000|000)': -0.022727272727272704, 'pabc(000|100)': -0.02272727272727271, 'pc(0|1)*pbc(00|00)': -0.022727272727272707, 'pa(0|1)*pc(0|0)*pc(0|1)': -0.0227272727272727, 'pc(0|0)*pabc(000|101)': 0.022727272727272697, 'pc(0|0)*pabc(000|001)': -0.02272727272727279, 'pc(0|1)*pabc(000|100)': 0.022727272727272704, 'pc(0|1)*pabc(000|000)': 0.0227272727272727}
dual_cert = {k: int(np.round(v/0.022727272727272707)) for k, v in dual_cert.items()}
print(dual_cert)
dual_cert_in_vars = sum(symbolic_moments[k]*v for k, v in dual_cert.items())
print(dual_cert_in_vars)

