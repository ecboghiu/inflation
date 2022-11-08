"""
This file contains helper functions to optimize (functions of) symbolic
variables under the constraint that the InflationSDP instance they implicitly
define must be feasible. Useful for exploring the set of parametrically-defined
distribution for which quantum inflation (at specific hierarchy levels) is
@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""
import numpy as np
import sympy as sp

from numbers import Real
from scipy.optimize import bisect, minimize, Bounds
from sympy.utilities.lambdify import lambdify
from typing import Callable, Dict, Tuple, Union

from inflation import InflationSDP
from inflation.sdp.quantum_tools import make_numerical
from inflation.sdp.monomial_classes import CompoundMonomial


def max_within_feasible(sdp: InflationSDP,
                        symbolic_values: Dict[CompoundMonomial, Callable],
                        method: str,
                        return_last_certificate=False,
                        **kwargs) -> Union[float,
                                           Tuple[float, sp.core.add.Add]]:
    """Maximize a single real variable within the set of feasible moment
    matrices determined by an ``InflationSDP``. The dependence of the moment
    matrices in the variable is specified by an assignment of monomials in the
    moment matrix to arbitrary expressions of the variable. This is useful for
    finding (bounds for) critical visibilities of distributions beyond which
    they are impossible to generate in a given quantum causal scenario.

    Parameters
    ----------
    sdp : InflationSDP
        The SDP problem under which to carry the optimization.
    symbolic_values : Dict[CompoundMonomial, Callable]
        The correspondence between monomials in the SDP problem and symbolic
        expressions depending on the variable to be optimized.
    method : str
        Technique used for optimization. Currently supported: ``"bisection"``
        for bisection algorithms, and ``"dual"`` for exploitation of the
        certificates of infeasibility (typically much fewer iteration steps).
    return_last_certificate : bool, optional
        Whether to return, along with the maximum value of the parameter, a
        separating surface that leaves the set of positive-semidefinite moment
        matrices in its positive side and evaluates to 0 in the maximum value
        reported.

    **kwargs
        Instructions on which extra symbolic values to assign numbers and
        options to be passed to the optimization routines (bounds, precision,
        tolerance, ...).

    Returns
    -------
    float
        The maximum value that the parameter can take under the set of
        positive-semidefinite moment matrices. This is the output when
        ``return_last_certificate=False``.
    Tuple[float, sympy.core.add.Add]
        The maximum value that the parameter can take under the set of
        positive-semidefinite moment matrices, and a corresponding separating
        surface (a root of the function corresponds to the critical feasible
        value of the parameter reported). This is the output when
        ``return_last_certificate=True``.
    """
    assert method in ["bisection", "dual"], \
        "Unknown optimization method. Please use \"bisection\" or \"dual\"."
    variables = set()
    for expr in symbolic_values.values():
        if not isinstance(expr, Real):
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
    """Implement the maximization of a variable within the feasible set of
    moment matrices using SciPy's bisection algorithm.


    Parameters
    ----------
    sdp : InflationSDP
        The SDP problem under which to carry the optimization.
    symbolic_values : Dict[CompoundMonomial, Callable]
        The correspondence between monomials in the SDP problem and symbolic
        expressions depending on the variable to be optimized.
    param : sympy.core.symbol.Symbol
        The variable to be optimized.

    **kwargs
        Additional arguments to ``sdp.set_values()`` and
        ``scipy.optimize.bisect()``.

    Returns
    -------
    float
        The maximum value that the parameter can take under the set of
        positive-semidefinite moment matrices. This is the output when
        ``return_last_certificate=False``.
    Tuple[float, sympy.core.add.Add]
        The maximum value that the parameter can take under the set of
        positive-semidefinite moment matrices, and a corresponding separating
        surface (a root of the function corresponds to the critical feasible
        value of the parameter reported). This is the output when
        ``return_last_certificate=True``.
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
    """Implement the maximization of a variable within the feasible set of
    moment matrices exploiting the certificates of infeasibility. For a given
    value of the parameter, a separating surface that leaves the set of
    positive-semidefinite moment matrices in its positive side is extracted.
    The next value of the parameter used is that which evaluates the surface to
    0.

    Parameters
    ----------
    sdp : InflationSDP
        The SDP problem under which to carry the optimization.
    symbolic_values : Dict[CompoundMonomial, Callable]
        The correspondence between monomials in the SDP problem and symbolic
        expressions depending on the variable to be optimized.
    param : sympy.core.symbol.Symbol
        The variable to be optimized.

    **kwargs
        Additional arguments to ``sdp.set_values()``, bounds of the
        optimization interval, precision of the optimization, verbosity and
        whether returning the last computed separating surface.

    Returns
    -------
    float
        The maximum value that the parameter can take under the set of
        positive-semidefinite moment matrices. This is the output when
        ``return_last_certificate=False``.
    Tuple[float, sympy.core.add.Add]
        The maximum value that the parameter can take under the set of
        positive-semidefinite moment matrices, and a corresponding separating
        surface (a root of the function corresponds to the critical feasible
        value of the parameter reported). This is the output when
        ``return_last_certificate=True``.
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
