"""
This file contains helper functions to optimize (functions of) symbolic
variables under the constraint that the InflationSDP instance they implicitly
define must be feasible. Useful for exploring the set of parametrically-defined
distribution for which quantum inflation (at specific hierarchy levels) is
@authors: Alejandro Pozas-Kerstjens, Elie Wolfe, Emanuel-Cristian Boghiu
"""
from typing import Callable
import numpy as np
from scipy.optimize import minimize
from sympy import var
from sympy.utilities.lambdify import lambdify

from causalinflation import InflationSDP
from causalinflation.quantum.general_tools import make_numerical


def symbolic_values_sdp_optimize(sdp: InflationSDP,
                                 list_of_symbols: list,
                                 names_to_expr_dict: dict,
                                 objective_to_min=(lambda x: -x[0]),
                                 x0=np.array([-1.0]),
                                 bounds=np.array([[0.0, 1.0]]),
                                 precision=0.0001,
                                 verbose=False) -> dict:
    """Minimizes a function of symbolic variables over the valuations of those
     variables which would make the SDP feasible, leveraging the infeasibility
    certificates yielded by the dual solution. To relate the symbolic
    variables to the monomials which appear in the SDP the user must supply a
    dictionary where the keys are names of monomial and the values are symbolic
    expressions for the corresponding monomial in terms of the symbolic
    variables. Optional bounds on the symbolic variables may be given.



    Parameters
    ----------
    sdp: InflationSDP
        The instance of InflationSDP upon which to iteratively call
        set_values() and solve().
    list_of_symbols: list
        A list of the symbolic variables in terms of which all
        symbolic expressions are defined in terms of.
    names_to_expr_dict: dict
        A dictionary where the keys are names of monomial and the values are
        symbolic expressions for the corresponding monomial in terms of the
        symbolic variables.
    objective_to_min: multivariate function, optional
        The function to minimize. If not specified this is taken to be equal to
        minus the value of the first symbolic variable in the list specified in
        the second argument.
    x0: numpy.ndarray, optional
        The initial numerical values to use for the symbolic variables. Should
        be chosen to make the SDP initially infeasible. When unspecified,
        defaults to assuming a single symbolic variable with an initial
        numerical value of 1.
    bounds: numpy.ndarray, optional
        An array up values pairs, indicating respectively the upper and lower
        numerical values to be considered for each of the symbolic variables.
        When unspecified, defaults to assuming a single symbolic variable with
        a numerical range between 0 and 1.
    precision: float, optional
        The iterative exploration of numerical values for the symbolic variables
        ceases if rejected range for the objective to be minimized does not
        increase by at least this value between iterations. If unspecified, the
        precision is set to 1e-4. May be set to zero.
    verbose: bool, optional
        If True, displays the updated numerical value of the function to be
        minimized every time solve() is called for the InflationSDP.

    Returns
    -------
        A dictionary where the key 'best_feasible_values' is associated with a
        numpy array of float which specify values for the symbolic variables
        which would make the SDP feasible (or which at least would not violate
        the final certificate). The key 'certificate' returns to most recent
        certificate, given in the form of a symbolic expression (a polynomial
        function of the symbolic variables). The key 'precision' return the
        precision value.
    """
    plural = ''
    if len(list_of_symbols) > 1:
        plural = 's'
    new_reject_vals = x0
    new_reject_funct_min = objective_to_min(new_reject_vals)
    old_reject_func_min = -np.inf
    old_reject_vals = new_reject_vals
    while new_reject_funct_min > old_reject_func_min + precision:
        old_reject_vals = new_reject_vals
        old_reject_func_min = new_reject_funct_min
        val_subs = dict(zip(list_of_symbols, new_reject_vals))
        sdp.solve(feas_as_optim=True,
                  solver_arguments={
                      "known_vars": make_numerical(names_to_expr_dict,
                                                   val_subs)})
        nonneg_expr = sum(names_to_expr_dict[k] * v
                          for k, v in
                          sdp.solution_object["dual_certificate"].items())
        nonneg_func = lambdify(list_of_symbols,
                               nonneg_expr,
                               modules='numpy', cse=True)
        constraints = ({'type': 'ineq', 'fun': nonneg_func})
        solution = minimize(objective_to_min, x0,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints,
                            options={'disp': False})
        new_reject_vals = solution['x']
        new_reject_funct_min = solution['fun']
        if verbose:
            if len(new_reject_vals) == 1:
                current_rejection = new_reject_vals[0]
            else:
                current_rejection = new_reject_vals
            print("Currently rejected value" + plural + ":", current_rejection)
    return {"best_feasible_values": old_reject_vals,
            "certificate": sdp.certificate_as_probs(),
            "precision": precision}


def max_such_that_feasible_via_dual(sdp: InflationSDP,
                                    dist_as_func: Callable,
                                    **kwargs) -> tuple:
    """
    Maximizes a single symbolic variable such that the distribution (resulting
    from applying a function to the value of the variable) makes the SDP
    feasible under set_distribution() and solve(). Internally uses certificates
    from the dual solutions to reject as large a swath as possible. Keyword
    arguments are passed forward to the 'symbolic_values_sdp_optimize' function.

    Parameters
    ----------
    sdp: InflationSDP
    dist_as_func: function
        A function which, given a scalar value, return a numpy array suitable
        for passing as an argument to sdp.set_distribution.

    Returns
    -------
    A tuple where index 0 gives the largest value that makes the SDP feasible,
    index 1 gives the certificates as symbolic polynomial expression, and
    index 2 gives the precision employed.
    """
    vis = var('v', real=True)
    sdp.set_distribution(dist_as_func(vis))
    sympolic_values = sdp._prepare_solver_arguments()["known_vars"]
    optimality_dict = symbolic_values_sdp_optimize(
        sdp=sdp,
        list_of_symbols=[vis],
        names_to_expr_dict=sympolic_values,
        **kwargs)
    return (optimality_dict["best_feasible_values"][0],
            optimality_dict["certificate"], optimality_dict["precision"])


def bisect(f: Callable,
           bounds=(0.0, 1.0),
           precision=1e-4,
           verbose=True,
           tolerance=0) -> float:
    """ Applies a bisection algorithm in order to find the largest value of a
    variable --- within some bounds --- which makes a function of the variable
    nonnegative. Its intended use is to maximize VISIBILITY, such that the
    function which should be nonnegative is the maximum value of the smallest
    eigenvalue of the moment matrix that can be realized with the given
    visibility is specified.


    Parameters
    ----------
    f: function
        A function of a single variable whose negativity indicates the rejection
        of the input value
    bounds: tuple, optional
        A specification of the range to explore for the target variable.
    precision: float, optional
        The bisection algorithm terminates if the step size drops below this
        value. By default, 1e-4; always set a strictly positive value.
    verbose: bool, optional
        If True (default), it will print the visibility and max of min
        eigenvalue at every iteration.
    tolerance: float, optional
        Treats the function as nonnegative so long as it returns a value
        greater than -abs(tolerance).
    Returns
    -------
        The largest scalar value in the specified range (up to the specified
        precision) which makes the function nonnegative (within the tolerance),
        or negative infinity if the lower bound makes the function negative.
    """
    threshhold = -np.abs(tolerance)
    (lo, up) = bounds
    x = lo
    step_size = up - lo
    while step_size > precision and lo <= x <= up:
        fx = f(x)
        if fx >= threshhold:
            x += step_size
        else:
            x -= step_size
        step_size /= 2
        if verbose:
            print(f'Maximum smallest eigenvalue: {fx:10.4g}   '
                  + f'Visibility = {x:.4g}')
    if x < lo:
        return -np.inf
    if x < up:
        return up
    return x


def max_such_that_feasible_via_bisect(sdp: InflationSDP,
                                      dist_as_func: Callable,
                                      **kwargs) -> float:
    """
    Maximizes a single symbolic variable such that the distribution (resulting
    from applying a function to the value of the variable) makes the SDP
    feasible under set_distribution() and solve(). Internally uses a bisection
    algorithm, increasing the value if the SDP is feasible and decreasing
    the value if the SDP is infeasible. Keyword arguments are passed forward to
    the 'bisect' function.

    Parameters
    ----------
    sdp: InflationSDP
    dist_as_func: function
        A function which, given a scalar value, return a numpy array suitable
        for passing as an argument to sdp.set_distribution.

    Returns
    -------
    The largest value (within the given range) which makes the SDP feasible, or
    negative infinity if the SDP cannot be made feasible in that range.
    """
    def f(vis):
        sdp.set_distribution(dist_as_func(vis), use_lpi_constraints=True)
        sdp.solve(feas_as_optim=True)
        return sdp.objective_value
    return bisect(f=f, **kwargs)
