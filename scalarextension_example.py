from inflation import InflationProblem, InflationSDP
import scipy
from scipy.optimize import bisect
import numpy as np
import sympy as sp
import cvxpy as cp

tri_line = InflationProblem(dag={"rho_AB": ["A", "B"],
                                 "rho_BC": ["B", "C"]},
                            outcomes_per_party=(2, 2, 2),
                            settings_per_party=(2, 2, 2),
                            classical_sources='all')

def P_2PR(vis=1):
    p = np.zeros((2, 2, 2, 2, 2, 2))
    for a, b, c, x, y, z in np.ndindex(*p.shape):
        p[a, b, c, x, y, z] = \
            (1 + vis * (-1) ** (a + b + c + x*y + y*z)) / 8
    return p

def min_eigenvalue(vis):
    sdp.set_distribution(P_2PR(vis), use_lpi_constraints=True)
    sdp.solve(feas_as_optim=True)
    print(f"Visibility = {vis:<6.4g}   " +
          "Maximum smallest eigenvalue: " +
          "{:10.4g}".format(sdp.objective_value))
    return sdp.objective_value


print("\n WITHOUT scalar extension\n")

sdp = InflationSDP(tri_line)
sdp.generate_relaxation("npa3")
sdp.set_distribution(P_2PR(1), use_lpi_constraints=True)
bisect(min_eigenvalue, 0, 1, xtol=1e-4)

print("\n WITH scalar extension\n")

sdp = InflationSDP(tri_line)
sdp.generate_relaxation("npa3")
###
sdp.relax_nonlinear_constraints(use_higherorder_inflation_terms=False)
###
sdp.set_distribution(P_2PR(1), use_lpi_constraints=True)
bisect(min_eigenvalue, 0, 1, xtol=1e-4)

print(sdp.moments)