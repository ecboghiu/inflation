import numpy as np


def PR_box_with_visibility(a, b, x, y, v):
    return (1 + v * (-1) ** (a + b + x * y)) / 4


def Anti_PR_box_with_visibility(a, b, x, y, v):
    return (1 - v * (-1) ** (a + b + x * y)) / 4


# for (x,y,a,b) in np.ndindex((2, 2, 2, 2)):
#     print(f"P({a},{b}|{x},{y}) = {Anti_PR_box_with_visibility(a, b, x, y, 1)}")
def NSBoxType1(v):
    tripartite_box = np.zeros((2, 2, 2, 2, 1, 2), dtype=object)
    for (x, y, a, b, c) in np.ndindex((2, 2, 2, 2, 2)):
        tripartite_box[a, b, c, x, 0, y] = 1 / 2 * PR_box_with_visibility(a, b,
                                                                          x, c,
                                                                          v)
    return tripartite_box


def NSBoxType2(v, post_selection_probability=1 / 2):
    tripartite_box = np.zeros((2, 2, 2, 2, 1, 2), dtype=object)
    white_noise_box = np.ones((2, 2, 2, 2), dtype=object) / 4
    for (x, y, a, c) in np.ndindex((2, 2, 2, 2)):
        tripartite_box[
            a, 0, c, x, 0, y] = post_selection_probability * PR_box_with_visibility(
            a, c, x, y, v)
    tripartite_box[:, 1, :, :, 0, :] = white_noise_box - tripartite_box[:, 0,
                                                         :, :, 0, :]
    return tripartite_box


# def NSBoxType2b(v, post_selection_probability=1/2):
#     tripartite_box = np.zeros((2,2,2,2,1,2), dtype=object)
#     white_noise_box = np.ones((2,2,2,2), dtype=object)/4
#     for (x,y,a,c) in np.ndindex((2, 2, 2, 2)):
#         tripartite_box[a, 1, c, x, 0, y] = post_selection_probability * Anti_PR_box_with_visibility(a, c, x, y, v)
#     tripartite_box[:, 0, :, :, 0, :] = white_noise_box - tripartite_box[:, 1, :, :, 0, :]
#     return tripartite_box

def NSBoxHalfAndHalf(v, post_selection_probability=1 / 2):
    return NSBoxType1(v) / 2 + NSBoxType2(v,
                                          post_selection_probability=post_selection_probability) / 2


from inflation import InflationProblem, InflationSDP, InflationLP

simplest_bilocal_scenario = InflationProblem({
    "lambda": ["a", "b"],
    "mu": ["b", "c"]},
    outcomes_per_party=[2, 2, 2],
    settings_per_party=[2, 1, 2],
    inflation_level_per_source=[2, 2],
    order=['a', 'b', 'c'],
    verbose=1)
# quantum_bilocal_SDP = InflationSDP(simplest_bilocal_scenario)
# quantum_bilocal_SDP.generate_relaxation("physical")
# quantum_bilocal_SDP.set_distribution(NSBoxHalfAndHalf(v=1, post_selection_probability=1/2))
# solution_raw = quantum_bilocal_SDP.solve(feas_as_optim=False)
# print(f"Quantum inflation report: {solution_raw.status}")
#
from sympy import Symbol
from inflation.sdp.optimization_utils import max_within_feasible
import mosek

simplest_bilocal_lp = InflationLP(simplest_bilocal_scenario,
                                  nonfanout=False,
                                  verbose=0)

print("TESTING NOISY PR-MIXTURE")
simplest_bilocal_lp.set_distribution(
    NSBoxHalfAndHalf(v=Symbol("v"), post_selection_probability=1 / 2))
symbolic_moments = simplest_bilocal_lp.known_moments.copy()
v, cert = max_within_feasible(simplest_bilocal_lp,
                              symbolic_moments,
                              method="dual",
                              return_last_certificate=True,
                              verbose=1,
                              solve_dual=True,
                              use_lpi_constraints=False)
print(f"Critical visibility via dual cert: {v}: via \n{cert}")

print("\n")
print("TESTING NOISY TSIRELSON-BOX-MIXTURE")
simplest_bilocal_lp.set_distribution(
    NSBoxHalfAndHalf(Symbol("v"), post_selection_probability=1 / 4))
symbolic_moments = simplest_bilocal_lp.known_moments.copy()
v, cert = max_within_feasible(simplest_bilocal_lp,
                              symbolic_moments,
                              method="dual",
                              return_last_certificate=True,
                              verbose=1,
                              solve_dual=True,
                              use_lpi_constraints=False,
                              # solverparameters={mosek.iparam.optimizer: mosek.optimizertype.dual_simplex}
                              )
print(f"Critical visibility via dual cert: {v}: via \n{cert}")
