"""
This file contains auxiliary functions of general purpose
@authors: Emanuel-Cristian Boghiu, Elie Wolfe and Alejandro Pozas-Kerstjens
"""
import numpy as np
from itertools import chain
from typing import Callable, Iterable


def blank_tqdm(*args, **kwargs):
    """Placeholder in case the tqdm library is not installed (see
    https://tqdm.github.io). If code is set to print a tqdm progress bar and
    tqdm is not installed, this just prints the ``desc`` argument.
    """
    try:
        if not kwargs["disable"]:
            print(kwargs["desc"])
    except KeyError:
        pass
    return args[0]


def flatten(nested):
    """Keeps flattening a nested lists of lists until  the
    first element of the resulting list is not a list.
    """
    if isinstance(nested, np.ndarray):
        return nested.ravel().tolist()
    else:
        while isinstance(nested[0], Iterable):
            nested = list(chain.from_iterable(nested))
        return nested


###############################################################################
# ROUTINES USED FOR OPTIMIZATION                                              #
###############################################################################
def bisect(f: Callable[[float], float],
           x0: float,
           x1: float,
           eps=1e-4,
           verbose=False) -> float:
    r"""Find a value :math:`x\in[x0, x1]` where :math:`f(x)=0` using the
    bisection method.

    Parameters
    ----------
    f : function
        The function to be evaluated. It must be a function from the real line
        into the real line.
    x0: float
        The lower end of the search interval.
    x1: float
        The higher end of the search interval.
    eps : float, optional
        The stopping criterion, expressed as the width of the interval centered
        in the output where the real solution is. By default ``1e-4``.
    verbose: bool, optional
        Whether information about each iteration is printed. By default
        ``False``.
    """
    assert f(x0)*f(x1) < 0., \
        "The function should have different signs in x0 and x1."
    x = (x0 + x1) / 2
    while abs(x1 - x0) > eps:
        fx = f(x)
        if fx >= 0.:
            x0 = x
        else:
            x1 = x
        if verbose:
            print(f"{f.__name__}({x:.4g}) = {fx:10.4g}")
        x = (x0 + x1) / 2
    return x
