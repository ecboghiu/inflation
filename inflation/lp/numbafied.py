"""
This file contains helper functions to which can be accelerated by JIT
compilation in numba.
@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""
import numpy as np

try:
    from numba import jit, prange, bool_
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
    for i in prange(length):
        standard_op = lexorder[i]
        for op in mon:
            if np.array_equal(op, standard_op):
                in_lex[i] = True
                break
    return in_lex

@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_apply_lexorder_perm_to_lexboolvecs(monomials_as_lexboolvecs: np.ndarray,
                                          lexorder_perms: np.ndarray) -> np.ndarray:
    orbits = np.zeros(len(monomials_as_lexboolvecs), dtype=int_) - 1
    for i, default_lexboolvec in enumerate(monomials_as_lexboolvecs):
        if orbits[i] == -1:
            for perm in lexorder_perms:
                negated_xor = np.logical_not(np.logical_xor(
                    default_lexboolvec[np.newaxis, perm],
                    monomials_as_lexboolvecs).sum(axis=1))
                orbits[negated_xor] = i
    return orbits

@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_outer_bitwise_or(a: np.ndarray, b: np.ndarray):
    a_adj = a[:, np.newaxis]
    b_adj = b.reshape((1,)+b.shape)
    temp = np.logical_or(a_adj, b_adj)
    return temp.reshape((-1, *temp.shape[2:]))

@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_outer_bitwise_xor(a: np.ndarray, b: np.ndarray):
    a_adj = a[:, np.newaxis]
    b_adj = b.reshape((1,)+b.shape)
    temp = np.logical_xor(a_adj, b_adj)
    return temp.reshape((-1, *temp.shape[2:]))