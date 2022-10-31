"""
This file contains helper functions to optimize (functions of) symbolic
variables under the constraint that the InflationSDP instance they implicitly
define must be feasible. Useful for exploring the set of parametrically-defined
distribution for which quantum inflation (at specific hierarchy levels) is
@authors: Alejandro Pozas-Kerstjens, Elie Wolfe, Emanuel-Cristian Boghiu
"""
from typing import Callable
import numpy as np
from scipy.optimize import minimize, Bounds
from sympy import var
from sympy.utilities.lambdify import lambdify

from causalinflation import InflationSDP
from causalinflation.quantum.general_tools import make_numerical


def max_such_that_feasible_via_dual(sdp: InflationSDP,
                                    dist_as_func: Callable,
                                    bounds=np.array([0.0, 1.0]),
                                    precision=0.0001,
                                    verbose=False) -> tuple:
    """
    Maximizes visibility such that the distribution (resulting
    from applying a function to the value of the visibility) makes the SDP
    feasible under set_distribution() and solve(). Internally uses certificates
    from the dual solutions to reject as large a swath as possible.

    Parameters
    ----------
    sdp: InflationSDP
    dist_as_func: function
        A function which, given a scalar value for the visibility, returns a
        numpy array suitable for passing as an argument to `set_distribution`.
    bounds: numpy.ndarray, optional
        An array of two values, indicating respectively the upper and lower
        numerical values to be considered for the visibility.
        When unspecified, defaults to a numerical range between 0 and 1.
    precision: float, optional
        The iterative exploration of numerical values for the visibility
        ceases if the rejected visibility range does not
        shrink by at least this value between iterations. If unspecified, the
        precision is set to 1e-4. May be set to zero.
    verbose: bool, optional
        If True, displays the updated numerical value of visibility every time
         `solve()` is called for the InflationSDP.

    Returns
    -------
    A tuple where index 0 gives the largest visibility value that makes the
    SDP feasible, and index 1 gives the certificates as a symbolic polynomial
    expression.
    """
    vis = var('v', real=True)
    func_to_minimize = lambdify([vis], -vis, modules='numpy', cse=True)
    sdp.set_distribution(dist_as_func(vis))
    names_to_expr_dict = sdp._prepare_solver_arguments()["known_vars"]
    bounds_for_scipy = Bounds(lb=bounds[0], ub=bounds[1])
    x0 = np.array([bounds[0]])
    new_reject_val = 1
    old_reject_val = np.inf
    discovered_certificates = []
    while new_reject_val < old_reject_val - precision:
        old_reject_val = new_reject_val
        val_sub = {vis: old_reject_val}
        new_known_vars = make_numerical(names_to_expr_dict, val_sub)
        sdp.solve(feas_as_optim=True,
                  solver_arguments={"known_vars": new_known_vars})
        discovered_certificates.append(sdp.certificate_as_probs())
        nonneg_expr = sum(names_to_expr_dict[k] * v
                          for k, v in
                          sdp.solution_object["dual_certificate"].items())
        nonneg_func = lambdify([vis],
                               nonneg_expr,
                               modules='numpy', cse=True)

        constraints = {'type': 'ineq', 'fun': nonneg_func}
        solution = minimize(func_to_minimize, x0,
                            method='SLSQP',
                            bounds=bounds_for_scipy,
                            constraints=constraints,
                            options={'disp': False})
        new_reject_val = solution['x'][0]
        if verbose:
            print("Currently rejected value:", new_reject_val)
    return min(new_reject_val, old_reject_val), discovered_certificates[-1]


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
        A function of a single variable whose negativity indicates the
        rejection of the input value
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
