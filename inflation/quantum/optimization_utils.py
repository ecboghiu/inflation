"""
This file contains helper functions to optimize (functions of) symbolic
variables under the constraint that the InflationSDP instance they implicitly
define must be feasible. Useful for exploring the set of parametrically-defined
distribution for which quantum inflation (at specific hierarchy levels) is
@authors: Alejandro Pozas-Kerstjens, Elie Wolfe, Emanuel-Cristian Boghiu
"""
import numpy as np
import sympy as sp

from scipy.optimize import bisect, minimize, Bounds
from sympy import Float
from sympy.utilities.lambdify import lambdify
from typing import Callable, Dict, Tuple, Union

from causalinflation import InflationSDP
from causalinflation.quantum.general_tools import make_numerical
from causalinflation.quantum.monomial_classes import CompoundMonomial


def max_within_feasible(sdp: InflationSDP,
                        symbolic_values: Dict[CompoundMonomial, Callable],
                        method: str,
                        return_last_certificate=False,
                        **kwargs):
    """Docs
    """
    assert method in ["bisection", "dual"], \
        "Unknown optimization method. Please use \"bisection\" or \"dual\"."
    variables = set()
    for expr in symbolic_values.values():
        if type(expr) not in [Float, float]:
            variables.update(expr.free_symbols)
    assert len(variables) == 1, \
        "Only optimization of a single variable is supported"
    param = variables.pop()
    kwargs.update({"return_last_certificate": return_last_certificate})

    if method == "bisection":
        return _maximize_via_bisect(sdp, symbolic_values, param, **kwargs)
    elif method == "dual":
        return _maximize_via_dual(sdp, symbolic_values, param, **kwargs)


###############################################################################
# OPTIMIZATION METHODS                                                        #
###############################################################################
def _maximize_via_bisect(sdp: InflationSDP,
                         symbolic_values: Dict[CompoundMonomial, Callable],
                         param: sp.core.symbol.Symbol,
                         **kwargs) -> Union[float,
                                            Tuple[float, sp.core.add.Add]]:
    """Maximizes a single symbolic variable such that the distribution (resulting
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
    bounds         = kwargs.get("bounds", np.array([0.0, 1.0]))
    only_specified = kwargs.get("only_specified_values", False)
    return_last    = kwargs.get("return_last_certificate", False)
    use_lpi        = kwargs.get("use_lpi_constraints", False)
    verbose        = kwargs.get("verbose", False)
    # Prepare bisect kwargs
    bisect_kwargs = {}
    for kwarg in ["args", "rtol", "maxiter", "full_output", "disp"]:
        try:
            bisect_kwargs[kwarg] = kwargs[kwarg]
        except Exception:
            pass
    try:
        bisect_kwargs["xtol"] = kwargs["xtol"]
    except Exception:
        bisect_kwargs["xtol"] = kwargs.get("precision", 1e-4)

    def f(value):
        evaluated_values = make_numerical(symbolic_values, {param: value})
        sdp.set_values(evaluated_values, use_lpi, only_specified)
        sdp.solve(feas_as_optim=True)
        if verbose:
            print(f"Parameter = {value:<6.4g}   " +
                  f"Maximum smallest eigenvalue: {sdp.objective_value:10.4g}")
        return sdp.objective_value
    crit_param = bisect(f, bounds[0], bounds[1], **bisect_kwargs)
    if return_last:
        return crit_param, sdp.certificate_as_probs()
    else:
        return crit_param


def _maximize_via_dual(sdp: InflationSDP,
                       symbolic_values: Dict[CompoundMonomial, Callable],
                       param: sp.core.symbol.Symbol,
                       **kwargs) -> Union[float,
                                          Tuple[float, sp.core.add.Add]]:
    """Maximizes visibility such that the distribution (resulting
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
    bounds         = kwargs.get("bounds", np.array([0.0, 1.0]))
    only_specified = kwargs.get("only_specified_values", False)
    precision      = kwargs.get("precision", 1e-4)
    return_last    = kwargs.get("return_last_certificate", False)
    use_lpi        = kwargs.get("use_lpi_constraints", False)
    verbose        = kwargs.get("verbose", False)

    symbol_names = {str(key): val for key, val in symbolic_values.items()}
    func_to_minimize = lambdify([param], -param, modules='numpy', cse=True)
    sp_bounds = Bounds(lb=bounds[0], ub=bounds[1])
    x0 = np.array([bounds[0]])
    new_ub = bounds[1]
    old_ub = np.inf
    discovered_certificates = []
    # Get a certificate for a value, find the critical value for it (that which
    # makes the certificate zero), and repeat with this new value
    while new_ub < old_ub - precision:
        old_ub = new_ub
        evaluated_values = make_numerical(symbolic_values, {param: new_ub})
        sdp.set_values(evaluated_values, use_lpi, only_specified)
        sdp.solve(feas_as_optim=True)
        discovered_certificates.append(sdp.certificate_as_probs())
        # The feasible region is where the certificate is positive
        nonneg_expr = sum(symbol_names[var] * coeff
                          for var, coeff in
                          sdp.solution_object["dual_certificate"].items())
        nonneg_func = lambdify([param],
                               nonneg_expr,
                               modules="numpy",
                               cse=True)
        constraints = {"type": "ineq", "fun": nonneg_func}
        solution = minimize(func_to_minimize, x0,
                            method='SLSQP',
                            bounds=sp_bounds,
                            constraints=constraints,
                            options={'disp': False})
        new_ub = solution['x'][0]
        if verbose:
            print("Current critical value:", new_ub)
    crit_param = min(new_ub, old_ub)
    if return_last:
        return crit_param, discovered_certificates[-1]
    else:
        return crit_param
