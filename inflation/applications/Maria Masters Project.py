import numpy as np
def PR_box_with_visibility(a, b, x, y, v):
    return (1 + v * (-1) ** (a + b + x * y)) / 4
def Anti_PR_box_with_visibility(a, b, x, y, v):
    return (1 - v * (-1) ** (a + b + x * y)) / 4

# for (x,y,a,b) in np.ndindex((2, 2, 2, 2)):
#     print(f"P({a},{b}|{x},{y}) = {Anti_PR_box_with_visibility(a, b, x, y, 1)}")

NSBoxType1 = np.zeros((2,2,2,2,1,2), dtype=float)
for (x,y,a,b,c) in np.ndindex((2, 2, 2, 2, 2)):
    NSBoxType1[a,b,c,x,0,y] = 1/2 *  PR_box_with_visibility(a, b, x, c, 1)

NSBoxType2 = np.zeros((2,2,2,2,1,2), dtype=float)
for (x,y,a,b,c) in np.ndindex((2, 2, 2, 2, 2)):
    if b==0:
        NSBoxType2[a, b, c, x, 0, y] = 1 / 2 * PR_box_with_visibility(a, c, x, y, 1)
    if b==1:
        NSBoxType2[a, b, c, x, 0, y] = 1 / 2 * Anti_PR_box_with_visibility(a, c, x, y, 1)

NSBoxHalfAndHalf = NSBoxType1/2 + NSBoxType2/2

from inflation import InflationProblem, InflationSDP
simplest_bilocal_scenario = InflationProblem({
    "lambda": ["a", "b"],
    "mu": ["b", "c"]},
    outcomes_per_party=[2, 2, 2],
    settings_per_party=[2, 1, 2],
    inflation_level_per_source=[2, 2],
    order=['a', 'b', 'c'],
    verbose=2)
quantum_bilocal_SDP = InflationSDP(simplest_bilocal_scenario)
quantum_bilocal_SDP.generate_relaxation("physical")
quantum_bilocal_SDP.set_distribution(NSBoxHalfAndHalf)
solution = quantum_bilocal_SDP.solve(feas_as_optim=True)

