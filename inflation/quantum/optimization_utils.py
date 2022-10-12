from typing import Callable

import numpy as np
from scipy.optimize import minimize
from sympy import var
from sympy.utilities.lambdify import lambdify

from causalinflation import InflationSDP
from causalinflation.quantum.general_tools import make_numerical


def symbolic_values_sdp_optimize(sdp: InflationSDP,
                                 list_of_symbols: list,
                                 dict_mon_names_to_symbolic_expressions: dict,
                                 objective_to_minimize_subject_to_sdp_feasibility=(lambda x0: -x0[0]),
                                 x0=np.array([1.0]),
                                 bounds=np.array([[0.0, 1.0]]),
                                 tolerance=0.0001,
                                 verbose=False):
    plural = ''
    if len(list_of_symbols) > 1:
        plural = 's'
    currently_rejected_values = x0
    currently_rejected_optimum = objective_to_minimize_subject_to_sdp_feasibility(currently_rejected_values)
    previously_rejected_optimum = -np.inf
    previously_rejected_values = currently_rejected_values
    while currently_rejected_optimum > previously_rejected_optimum + tolerance:
        previously_rejected_values = currently_rejected_values
        previously_rejected_optimum = currently_rejected_optimum
        numerical_substitutions = dict(zip(list_of_symbols, currently_rejected_values))
        sdp.solve(feas_as_optim=True,
                  solver_arguments={"known_vars": make_numerical(dict_mon_names_to_symbolic_expressions,
                                                                 numerical_substitutions)})
        expression_which_must_nonnegative = sum(dict_mon_names_to_symbolic_expressions[k] * v
                                                for k, v in sdp.solution_object["dual_certificate"].items())
        nonnegativity_constraint_from_sdp_cert = lambdify(list_of_symbols,
                                                          expression_which_must_nonnegative,
                                                          modules='numpy', cse=True)
        constraints = ({'type': 'ineq', 'fun': nonnegativity_constraint_from_sdp_cert})
        solution = minimize(objective_to_minimize_subject_to_sdp_feasibility, x0,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints,
                            options={'disp': False})
        currently_rejected_values = solution['x']
        currently_rejected_optimum = solution['fun']
        if verbose:
            if len(currently_rejected_values) == 1:
                current_rejection = currently_rejected_values[0]
            else:
                current_rejection = currently_rejected_values
            print("Currently rejected value" + plural + ":", current_rejection)
    return {"best_feasible_values": previously_rejected_values,
            "tolerance": tolerance,
            "certificate": sdp.certificate_as_probs()}


def maximize_scalar_such_that_sdp_feasible_using_dual_solution(sdp: InflationSDP,
                                                               distribution_as_scalar_function: Callable,
                                                               tolerance=0.0001,
                                                               **kwargs) -> tuple:
    vis = var('v', real=True)
    sdp.set_distribution(distribution_as_scalar_function(vis))
    sympolic_values = sdp._prepare_solver_arguments()["known_vars"]
    optimality_dict = symbolic_values_sdp_optimize(sdp=sdp,
                                                   list_of_symbols=[vis],
                                                   dict_mon_names_to_symbolic_expressions=sympolic_values,
                                                   tolerance=tolerance,
                                                   **kwargs)
    return optimality_dict["best_feasible_values"][0], optimality_dict["certificate"], tolerance

def bisect(f, x0, x1, eps=1e-4, verbose=1):
    # We assume f(x1) < 0 and f(x0) > 0
    x = (x0 + x1) / 2
    while abs(x1 - x0) > eps:
        fx = f(x)
        if fx >= 0:
            x0 = x
        else:
            x1 = x
        if verbose:
            print(f'Maximum smallest eigenvalue: {fx:10.4g}   Visibility = {x:.4g}')
        x = (x0 + x1) / 2
    return x

def maximize_scalar_such_that_sdp_feasible_using_bisection(sdp: InflationSDP,
                                                           distribution_as_scalar_function: Callable,
                                                           bounds=(0, 1),
                                                           tolerance=0.0001,
                                                           verbose=True) -> float:
    def f(vis):
        sdp.set_distribution(distribution_as_scalar_function(vis), use_lpi_constraints=True)
        sdp.solve(feas_as_optim=True)
        return sdp.objective_value

    return bisect(f, bounds[0], bounds[1], tolerance, verbose=verbose)
