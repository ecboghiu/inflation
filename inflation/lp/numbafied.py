"""
This file contains helper functions to which can be accelerated by JIT
compilation in numba.
@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""
import numpy as np

try:
    from numba import jit, bool_
    from numba.types import int_
    nopython = True
except ImportError:
    print("Warning: NOT using Numba for JIT compilation.")
    def jit(*args, **kwargs):
        return lambda f: f
    bool_    = bool
    int_     = int
    nopython = False
    prange   = range
cache    = True
if not nopython:
    bool_  = bool
    int_ = int
from sparse import GCXS

@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_mon_to_lexrepr_bool(mon: np.ndarray,
                           lexorder: np.ndarray) -> np.array:
    """Convert a monomial to its lexicographic representation, in the form of
    an array of integers representing the lexicographic rank of each operator.

    Parameters
    ----------
    mon : numpy.ndarray
        Input monomial in 2d array format.
    lexorder : numpy.ndarray
        Matrix with rows as operators where the index of the row gives
        the lexicographic order of the operator.

    Returns
    -------
    np.array
        Boolean vector of length lexorder, with 1 if something in mon is in
        lexorder
    """
    length = lexorder.shape[0]
    in_lex = np.zeros(length, dtype=bool_)
    for i in range(length):
        standard_op = lexorder[i]
        for op in mon:
            if np.array_equal(op, standard_op):
                in_lex[i] = True
                break
    return in_lex

def nb_outer_bitwise_or(a: np.ndarray, b: np.ndarray):
    if type(a) == np.ndarray:
        a = GCXS(a)
    if type(b) == np.ndarray:
        b = GCXS(b)
    a_adj = a.reshape((a.shape[0], 1) + a.shape[1:])
    b_adj = b.reshape((1, ) + b.shape)
    temp = a_adj + b_adj
    return temp.reshape((-1, *temp.shape[2:]))#.todense()


# @jit(nopython=nopython, cache=cache, forceobj=not nopython)
# def nb_outer_bitwise_or(a: np.ndarray, b: np.ndarray):
#     print('------------')
#     print('a', a)
#     print('b', b)
#     a_adj = np.expand_dims(a, 1)
#     b_adj = b.reshape((1,)+b.shape)
#     temp = np.logical_or(a_adj, b_adj)
#     return temp.reshape((-1, *temp.shape[2:]))

@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_outer_bitwise_xor(a: np.ndarray, b: np.ndarray):
    a_adj = np.expand_dims(a, 1)
    b_adj = b.reshape((1,)+b.shape)
    temp = np.logical_xor(a_adj, b_adj)
    return temp.reshape((-1, *temp.shape[2:]))

@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_is_do_conditional(monomial: np.ndarray) -> bool_:
    """Determine whether a given atomic monomial admits an identification with
    a do conditional of the original scenario.

    Parameters
    ----------
    monomial : np.ndarray
        List of operators, denoted each by a list of indices

    Returns
    -------
    bool
        Whether the monomial is knowable or not.
    """
    if len(monomial) <= 1:
        return True
    # Mappable monomials have at most one copy of each source in the DAG
    for source in monomial.T[1:-2]:
        if len(np.unique(source[np.flatnonzero(source)])) > 1:
            return False
    return True