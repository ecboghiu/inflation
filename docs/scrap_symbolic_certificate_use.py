from causalinflation import InflationProblem, InflationSDP
import numpy as np
from sympy import var
# from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.lambdify import lambdify
# from sympy.utilities.autowrap import ufuncify
from scipy.optimize import minimize

def symbolic_values_sdp_optimize(sdp: InflationSDP,
                                 vars: list,
                                 sympolic_values: dict,
                                 objective_to_minimize_subject_to_sdp_feasibility=(lambda v: -v),
                                 x0=np.array([0.0]),
                                 bounds=np.array([[0.0, 1.0]])):

    currently_rejected_values = bounds[:, -1]
    previously_rejected_values = currently_rejected_values+1
    currently_rejected_optimum = objective_to_minimize_subject_to_sdp_feasibility(*currently_rejected_values)
    previously_rejected_optimum = objective_to_minimize_subject_to_sdp_feasibility(*previously_rejected_values)
    while currently_rejected_optimum > previously_rejected_optimum + 0.001:  # adjustable tolerance
        numeric_values = dict()
        for k, v in sympolic_values.items():
            try:
                numeric_values[k] = float(v.evalf(subs=dict(zip(vars, currently_rejected_values))))
            except AttributeError:
                numeric_values[k] = float(v)
        sdp.solve(feas_as_optim=True,
                  core_solver_arguments={"known_vars": numeric_values})
        current_cert = sdp.solution_object["dual_certificate"]
        nonegativity_constraint_from_sdp_cert = lambdify(vars, sum(
            value * sympolic_values[name] for name, value in current_cert.items()),
                                                         modules='numpy', cse=True)
        constraints = ({'type': 'ineq', 'fun': nonegativity_constraint_from_sdp_cert})
        solution = minimize(objective_to_minimize_subject_to_sdp_feasibility, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                            options={'disp': True})
        currently_rejected_values = solution['x']
        previously_rejected_optimum = currently_rejected_optimum
        currently_rejected_optimum = solution['fun'][0]
        print("Currently rejected values:", currently_rejected_values)
        print("Currently rejected optimum:", currently_rejected_optimum)
        print(u'\u2500' * 80, sep='\n\n')



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

vis = var('v', real=True)
sdp.set_distribution(P_2PR(vis))
sympolic_values = sdp.prepare_solver_arguments()["known_vars"]

symbolic_values_sdp_optimize(sdp=sdp,
                             vars=[vis],
                             sympolic_values=sympolic_values)
