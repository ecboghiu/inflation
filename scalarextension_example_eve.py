from inflation import InflationProblem, InflationSDP
import scipy
from scipy.optimize import bisect
import numpy as np
import sympy as sp
import cvxpy as cp

evans_eavesdropped = InflationProblem(dag={"rhoABE": ["A", "B", "E"],
                                "rhoBCE": ["B", "C", "E"],
                                "B": ["A", "C", "E"]},
                           outcomes_per_party=(2, 2, 2, 2),
                           settings_per_party=(1, 1, 1, 1),
                           order=["A", "B", "C", "E"],
                           verbose=1)

# def P_2PR(vis=1):
#     p = np.zeros((2, 2, 2, 2, 2, 2))
#     for a, b, c, x, y, z in np.ndindex(*p.shape):
#         p[a, b, c, x, y, z] = \
#             (1 + vis * (-1) ** (a + b + c + x*y + y*z)) / 8
#     return p

# def min_eigenvalue(vis):
#     sdp.set_distribution(P_2PR(vis), use_lpi_constraints=True)
#     sdp.solve(feas_as_optim=True)
#     print(f"Visibility = {vis:<6.4g}   " +
#           "Maximum smallest eigenvalue: " +
#           "{:10.4g}".format(sdp.objective_value))
#     return sdp.objective_value


sdp = InflationSDP(evans_eavesdropped, verbose=1)
sdp.generate_relaxation("npa5")
###
sdp.relax_nonlinear_constraints()

meas = sdp.measurements
A0 = meas[0][0][0][0]
B0 = meas[1][0][0][0]
C0 = meas[2][0][0][0]
E0 = meas[3][0][0][0]


P = np.zeros((2, 2, 2, 1, 1, 1))
for a, c in np.ndindex(2, 2):
    P[a,0,c,0,0,0]=1/16*(1+(1/np.sqrt(2)*(-1)**(a+c)))
    P[a,1,c,0,0,0]=1/16*(3+(-1)**(a+c)+ (1/np.sqrt(2))*((-1)**c-(-1)**a))

values={ m.name : m.compute_marginal(P) for m in sdp.knowable_atoms if "E" not in m.name}

sdp.update_values(values, use_lpi_constraints=True)
sdp.set_objective(A0*E0 + (1-A0)*(1-E0))

sdp.solve(verbose=1)
print(sdp.objective_value)
