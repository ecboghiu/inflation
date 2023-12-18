"""
This file contains helper functions to which can be accelerated by JIT
compilation in numba.
@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""
import numpy as np

try:
    from numba import jit, prange, bool_
    from numba.types import int_
    from numba.typed import Dict, List, Set
    empty_dict = Dict.empty(key_type=int_, value_type=int_)
    nopython = True
except ImportError:
    def jit(*args, **kwargs):
        return lambda f: f
    bool_    = bool
    int_     = int
    nopython = False
    prange   = range
    empty_dict = dict()
    List = list
    Set = set
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
def bitvec_to_int(bitvec: np.ndarray) -> int_:
    val = 0
    for b in bitvec.flat:
        val = np.left_shift(val, 1)
        val = np.bitwise_or(val, b)
    return val


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_apply_lexorder_perm_to_lexboolvecs(monomials_as_lexboolvecs: np.ndarray,
                                          lexorder_perms: np.ndarray) -> np.ndarray:
    # Note: dictionary creation seems wasteful,
    # but this function is not called within loops.
    lookup_dict = empty_dict.copy()
    for i, bitvec in enumerate(monomials_as_lexboolvecs):
        lookup_dict[bitvec_to_int(bitvec)] = i
    lookup_dict_keys = Set(lookup_dict.keys())
    # lookup_dict = {bitvec_to_int(bitvec): i for i, bitvec in
    #                enumerate(monomials_as_lexboolvecs)}
    orbits = np.zeros(len(monomials_as_lexboolvecs), dtype=int_) - 1
    for i, default_lexboolvec in enumerate(monomials_as_lexboolvecs):
        if orbits[i] == -1:
            # alternative_lexboolvecs = default_lexboolvec[lexorder_perms]
            equivalent_monomial_positions = List()
            for perm in lexorder_perms:
                bitvec = default_lexboolvec[perm]
                bitvec_as_int = bitvec_to_int(bitvec)
                if bitvec_as_int in lookup_dict_keys:
                    equivalent_monomial_positions.append(lookup_dict[bitvec_as_int])
            orbits[equivalent_monomial_positions] = i
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