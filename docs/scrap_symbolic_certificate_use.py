from causalinflation import InflationProblem, InflationSDP
import numpy as np

qbilocal = InflationProblem(dag={'rho_AB': ['A', 'B'],
                                 'rho_BC': ['B', 'C']},
                            outcomes_per_party=(2, 2, 2),
                            settings_per_party=(2, 2, 2),
                            inflation_level_per_source=(2, 2))
sdp = InflationSDP(qbilocal)
sdp.generate_relaxation('npa2')

def P_2PR(vis=1):
    p = np.zeros((2, 2, 2, 2, 2, 2))
    for a, b, c, x, y, z in np.ndindex(*p.shape):
        p[a, b, c, x, y, z] = ( 1 + (-1) ** (a + b + c + x*y + y*z) ) / 8
    return vis * p + (1 - vis) * np.ones(p.shape) / 2 ** 3

from sympy import var
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.lambdify import lambdify
from scipy.optimize import minimize
vis = var('v', real=True)

sdp.set_distribution(P_2PR(vis))
sympolic_info = sdp.prepare_solver_arguments()["known_vars"]

previous_rejected_visibility = 2.0
currently_rejected_visibility = 1.0
while previous_rejected_visibility > currently_rejected_visibility + 0.001:  # adjustable tolerance

    sdp.set_distribution(P_2PR(currently_rejected_visibility))
    sdp.solve(feas_as_optim=True)
    current_cert = sdp.solution_object["dual_certificate"]
    objective = lambda v: -v
    nonegativity_constraint_from_sdp_cert = lambdify(vis, sum(value*sympolic_info[name] for name, value in current_cert.items()),
                                   modules='numpy', cse=True)
    constraints = ({'type': 'ineq', 'fun': nonegativity_constraint_from_sdp_cert})
    x0 = np.array([0.0])
    bounds = np.array([[0.0, 1.0]])
    solution = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                        options={'disp': True})
    previous_rejected_visibility = currently_rejected_visibility
    currently_rejected_visibility = solution['x'][0]
    print("Currently rejected visibility:", currently_rejected_visibility)
    print(u'\u2500' * 80, sep='\n\n')
