"""
This file contains helper functions to optimize (functions of) symbolic
variables under the constraint that the InflationLP or InflationSDP instance
they implicitly define must be feasible. Useful for exploring the set of
parametrically-defined distribution for which quantum inflation (at specific
hierarchy levels) is

@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""
import numpy as np
import sympy as sp

from numbers import Real
from scipy.optimize import bisect, minimize, Bounds
from sympy.utilities.lambdify import lambdify
from typing import Dict, Tuple, Union
from functools import lru_cache

from . import InflationLP, InflationSDP
from .utils import make_numerical
from .sdp.monomial_classes import CompoundMomentSDP


def max_within_feasible(program: Union[InflationLP, InflationSDP],
                        symbolic_values: Dict[CompoundMomentSDP,
                                              sp.core.expr.Expr],
                        method: str,
                        return_last_certificate=False,
                        **kwargs) -> Union[float,
                                           Tuple[float, dict]]:
    """Maximize a single real variable within the set of feasible moment
    matrices determined by an ``InflationSDP``. The dependence of the moment
    matrices in the variable is specified by an assignment of monomials in the
    moment matrix to arbitrary expressions of the variable. This is useful for
    finding (bounds for) critical visibilities of distributions beyond which
    they are impossible to generate in a given quantum causal scenario.

    Parameters
    ----------
    program : Union[InflationLP, InflationSDP]
        The problem under which to carry the optimization.
    symbolic_values : Dict[CompoundMomentSDP, Callable]
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
        distributions compatible with an inflation (for LPs) or under the set of
        positive-semidefinite moment matrices (for SDPs). This is the output
        when ``return_last_certificate=False``.
    Tuple[float, dict]
        The maximum value that the parameter can take under the set of
        distributions compatible with an inflation (for LPs) or under the set of
        positive-semidefinite moment matrices (for SDPs), and a corresponding
        separating surface (a root of the function corresponds to the critical
        feasible value of the parameter reported). This is the output when
        ``return_last_certificate=True``.
    """
    assert method in ["bisection", "dual"], \
        "Unknown optimization method. Please use \"bisection\" or \"dual\"."
    variables = set()
    for expr in symbolic_values.values():
        if not isinstance(expr, Real):
            variables.update(expr.atoms(sp.Symbol))
    assert len(variables) != 0, \
        "No variable to be optimized detected"
    assert len(variables) == 1, \
        "Only optimization of a single variable is supported"
    param = variables.pop()
    kwargs.update({"return_last_certificate": return_last_certificate})

    if method == "bisection":
        return _maximize_via_bisect(program, symbolic_values, param, **kwargs)
    elif method == "dual":
        return _maximize_via_dual(program, symbolic_values, param, **kwargs)


###############################################################################
# OPTIMIZATION METHODS                                                        #
###############################################################################
def _maximize_via_bisect(program: Union[InflationLP, InflationSDP],
                         symbolic_values: Dict[CompoundMomentSDP,
                                               sp.core.expr.Expr],
                         param: sp.core.symbol.Symbol,
                         **kwargs) -> Union[float,
                                            Tuple[float, dict]]:
    """Implement the maximization of a variable within the feasible set of
    distributions using SciPy's bisection algorithm.


    Parameters
    ----------
    program : Union[InflationLP, InflationSDP]
        The problem under which to carry the optimization.
    symbolic_values : Dict[CompoundMomentSDP, Callable]
        The correspondence between monomials in the problem and symbolic
        expressions depending on the variable to be optimized.
    param : sympy.core.symbol.Symbol
        The variable to be optimized.

    **kwargs
        Additional arguments to ``program.set_values()`` and
        ``scipy.optimize.bisect()``.

    Returns
    -------
    float
        The maximum value that the parameter can take under the set of
        distributions compatible with an inflation (for LPs) or under the set of
        positive-semidefinite moment matrices (for SDPs). This is the output
        when ``return_last_certificate=False``.
    Tuple[float, sympy.core.add.Add]
        The maximum value that the parameter can take under the set of
        distributions compatible with an inflation (for LPs) or under the set of
        positive-semidefinite moment matrices (for SDPs), and a corresponding
        separating surface (a root of the function corresponds to the critical
        feasible value of the parameter reported). This is the output when
        ``return_last_certificate=True``.
    """
    bounds         = kwargs.get("bounds", np.array([0.0, 1.0]))
    only_specified = kwargs.get("only_specified_values", False)
    return_last    = kwargs.get("return_last_certificate", False)
    precision      = kwargs.get("precision", 1e-4)
    use_lpi        = kwargs.get("use_lpi_constraints", False)
    verbose        = kwargs.get("verbose", False)
    solve_dual     = kwargs.get("solve_dual", True)
    solverparameters = kwargs.get("solverparameters", None)
    # Prepare bisect kwargs
    bisect_kwargs = {}
    for kwarg in ["args", "rtol", "maxiter", "full_output", "disp"]:
        try:
            bisect_kwargs[kwarg] = kwargs[kwarg]
        except KeyError:
            pass
    try:
        bisect_kwargs["xtol"] = kwargs["xtol"]
    except KeyError:
        bisect_kwargs["xtol"] = precision

    discovered_certificates = []

    @lru_cache(maxsize=2, typed=False)
    def f(value: float) -> float:
        evaluated_values = make_numerical(symbolic_values, {param: value})
        program.set_values(evaluated_values,
                       use_lpi_constraints=use_lpi,
                       only_specified_values=only_specified)
        program.solve(feas_as_optim=True,
                  solve_dual=solve_dual,
                  solverparameters=solverparameters)
        if verbose:
            print(f"Parameter = {value:<6.4g}   " +
                  f"Maximum smallest eigenvalue: {program.objective_value:10.4g}")
        discovered_certificate = program.certificate_as_dict()
        if len(discovered_certificate):  # Checking if certificate has any nonzero coefficients
            discovered_certificates.append((value, discovered_certificate))
        return program.objective_value+precision/2048
    assert f(bounds[1])+precision/2048 < 0, f"Certificate of infeasibility for v=1 is {f(bounds[1])}, but must be <{precision/2048}."
    assert f(bounds[0])+precision/2048 > 0, f"Certificate of feasibility for v=0 is {f(bounds[0])}, but must be nonnegative."
    crit_param = bisect(f, bounds[0], bounds[1], **bisect_kwargs, full_output=False)
    if return_last:
        return crit_param, discovered_certificates[-1][-1]
    else:
        return crit_param


def _maximize_via_dual(program: Union[InflationLP, InflationSDP],
                       symbolic_values: Dict[CompoundMomentSDP,
                                             sp.core.expr.Expr],
                       param: sp.core.symbol.Symbol,
                       **kwargs) -> Union[float,
                                          Tuple[float, dict]]:
    """Implement the maximization of a variable within the feasible set of
    distributions exploiting the certificates of infeasibility. For a given
    value of the parameter, a separating surface that leaves the set of
    fesible distributions in its positive side is extracted. The next value of
    the parameter used is that which evaluates the surface to 0.

    Parameters
    ----------
    program : Union[InflationLP, InflationSDP]
        The problem under which to carry the optimization.
    symbolic_values : Dict[CompoundMomentSDP, Callable]
        The correspondence between monomials in the problem and symbolic
        expressions depending on the variable to be optimized.
    param : sympy.core.symbol.Symbol
        The variable to be optimized.

    **kwargs
        Additional arguments to ``program.set_values()``, bounds of the
        optimization interval, precision of the optimization, verbosity and
        whether returning the last computed separating surface.

    Returns
    -------
    float
        The maximum value that the parameter can take under the set of
        distributions compatible with an inflation (for LPs) or under the set of
        positive-semidefinite moment matrices (for SDPs). This is the output
        when ``return_last_certificate=False``.
    Tuple[float, dict]
        The maximum value that the parameter can take under the set of
        distributions compatible with an inflation (for LPs) or under the set of
        positive-semidefinite moment matrices (for SDPs), and a corresponding
        separating surface (a root of the function corresponds to the critical
        feasible value of the parameter reported). This is the output when
        ``return_last_certificate=True``.
    """
    bounds         = kwargs.get("bounds", np.array([0.0, 1.0]))
    only_specified = kwargs.get("only_specified_values", False)
    precision      = kwargs.get("precision", 1e-4)
    return_last    = kwargs.get("return_last_certificate", False)
    use_lpi        = kwargs.get("use_lpi_constraints", False)
    verbose        = kwargs.get("verbose", False)
    solve_dual     = kwargs.get("solve_dual", True)
    solverparameters = kwargs.get("solverparameters", None)

    symbol_names = {str(key): val for key, val in symbolic_values.items()}
    func_to_minimize = lambdify([param], -param, modules='numpy', cse=True)
    sp_bounds = Bounds(lb=bounds[0], ub=bounds[1])
    x0 = np.array([bounds[0]])
    new_ub = bounds[1]-precision
    old_ub = np.inf
    discovered_certificates = []
    # Get a certificate for a value, find the critical value for it (that which
    # makes the certificate zero), and repeat with this new value
    while new_ub+precision < old_ub:
        old_ub = new_ub
        evaluated_values = make_numerical(symbolic_values,
                                          {param: new_ub+precision})
        program.set_values(evaluated_values,
                           use_lpi_constraints=use_lpi,
                           only_specified_values=only_specified)
        program.solve(feas_as_optim=True,
                      solve_dual=solve_dual,
                      solverparameters=solverparameters)

        # The feasible region is where the certificate is positive
        nonneg_expr = sum(symbol_names[var] * coeff
                          for var, coeff in
                          program.solution_object["dual_certificate"].items())
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
        discovered_certificates.append((new_ub , program.certificate_as_dict()))
        if verbose:
            print(f"Current critical value: {new_ub} (seeded by {old_ub+precision})")
    # assert len(discovered_certificates)>1, """
    # Critical error - optimization failed to yield a useful certificate
    # even when given the initial (infeasible) seed.
    # """

    crit_param = max(new_ub, old_ub)
    if return_last:
        return crit_param, discovered_certificates[-1][-1]
    else:
        return crit_param
