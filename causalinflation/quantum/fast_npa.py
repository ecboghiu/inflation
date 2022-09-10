"""
This file contains helper functions to manipulate monomials and generate moment
matrices. The functions in this file can be accelerated by JIT compilation in
numba.
@authors: Alejandro Pozas-Kerstjens, Emanuel-Cristian Boghiu
"""
import numpy as np
from scipy.sparse import dok_matrix  # , coo_matrix, coo

from causalinflation.quantum.types import List, Dict, Tuple, Union
#import causalinflation

# Had to insert this because of circular imports
# from collections import defaultdict, deque

##########################################
# Problems with cached functions with numba, while developing I recommend
# cleaning the cache, later we can remove this
import os

def kill_files(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print("failed on filepath: %s" % file_path)

def kill_numba_cache():

    root_folder = os.path.realpath(__file__ + "/../../")

    for root, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            if dirname == "__pycache__":
                try:
                    kill_files(root + "/" + dirname)
                except Exception as e:
                    print("failed on %s", root)

try:
    import numba
    from numba import jit
    from numba.types import bool_, void
    from numba.types import uint16 as uint16_
    from numba.types import int64 as int64_
    # from numba import types
    from numba.typed import Dict as nb_Dict
except ImportError:
    def jit(*args, **kwargs):
        return lambda f: f
    bool_   = bool
    uint16_ = np.uint16
    int64_  = np.int
    nb_Dict = dict
    void    = None
try:
    from tqdm import tqdm
except ImportError:
    from ..utils import blank_tqdm as tqdm

cache = True
nopython = True
if nopython == False:
    bool_ = bool
    uint16_ = np.uint16
    int64_ = np.int
    nb_Dict = dict

@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_op_eq_op(op1: np.ndarray, op2: np.ndarray) -> bool_:
    """Check if two operators are equal.

    NOTE: There is no check for shape consistency. As this is
    an internal function, it is assumed it is used correctly.
    """
    # if mon1.shape[0] != mon2.shape[0]: # TODO
    #     return False
    for i in range(op1.shape[0]):
        if op1[i] != op2[i]:
            return False
    return True
    #return np.array_equal(mon1, mon1)  # Slower when compiled with numba

@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_linsearch(arr: np.ndarray, value) -> int:
    """Return the index of the first element in arr that is equal to value
    or -1 if the element is not found."""
    for index in range(arr.shape[0]):
        if arr[index] == value:
            return index
    return -1

@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_unique(arr: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Find the unique elements in an array without sorting
    and in order of appearance.

    Parameters
    ----------
    arr : np.ndarray
        The array to search.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The unique values unsorted and their indices.

    Examples
    --------
    >>> nb_unique(np.array([1, 3, 3, 2, 2, 5, 4]))
    (array([1, 3, 2, 5, 4], dtype=int16), array([0, 1, 3, 5, 6], dtype=int16))
    """

    # One can use return_index=True with np.unique but I find this incompatible with numba so I do it by hand
    uniquevals = np.unique(arr)
    nr_uniquevals = uniquevals.shape[0]

    indices = np.zeros(nr_uniquevals).astype(int64_)
    for i in range(nr_uniquevals):
        indices[i] = nb_linsearch(arr, uniquevals[i])
    indices.sort()

    uniquevals_unsorted = np.zeros(nr_uniquevals).astype(int64_)
    for i in range(nr_uniquevals):
        # Undo the sorting done by np.unique()
        uniquevals_unsorted[i] = arr[indices[i]]

    return uniquevals_unsorted, indices

@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_op_lexorder(op: np.array, lexorder: np.ndarray) -> int:
    """Map each operator to a unique integer for hashing and lexicographic
    ordering comparison purposes.
    """
    # # summ = op[0]
    # # prod = 1
    # # for i in range(1, op.shape[0]):
    # #     prod *= NB_RADIX
    # #     summ += op[i]*prod
    # # return summ
    # Performance note: instead of having a matrix where the first column
    # stores the lexicographic order, we can have a matrix where the rows are
    # ordered in the custom lexico-graphical order and then we simply return
    # the row index where the row is equal to the input. Strangely enough,
    # that is 40% slower with Numba! It is a headscratcher why returning
    # i instead of lexorder[i, 0] is slower...
    for i in range(lexorder.shape[0]):
        if nb_op_eq_op(lexorder[i, :], op):
            return i

@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_mon_to_lexrepr(mon: np.ndarray, lexorder: np.ndarray) -> np.array:
    """Convert a monomial to its lexicographic representation, as an
    array of integers representing the lex rank of each operator."""
    # lex = np.zeros(mon.shape[0], dtype=mon.dtype)
    lex = np.zeros_like(mon[:, 0])
    for i in range(mon.shape[0]):
        lex[i] = nb_op_lexorder(mon[i], lexorder)
    return lex


# # Elie: This function does not work, nor is it used anywhere, so I'm commenting it out.
# @jit(nopython=nopython, cache=cache, forceobj=not nopython)
# def nb_lexrepr_to_mon(lexrepr: np.ndarray, lexorder: np.ndarray) -> np.array:
#     """Convert a monomial to its lexicographic representation, as an
#     array of integers representing the lex rank of each operator."""
#     mon = np.zeros(mon.shape, dtype=uint16_)
#     for i in range(lexrepr.shape[0]):
#         mon[i] = nb_op_lexorder(mon[i], lexorder)
#     return lexorder[lexrepr]

@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_sort_lexorder(op_lexorder: np.array) -> np.array:
    """Find the permutation that brings the operator to its lexicographic
    ordering. E.g.: [3, 1, 2, 0] -perm-> [0, 1, 2, 3] """
    perm = np.zeros(op_lexorder.shape[0], dtype=uint16_)
    for i in range(op_lexorder.shape[0]):
        perm[op_lexorder[i]] = i
    return perm

# @jit(nopython=nopython, cache=cache, forceobj=not nopython)
# def nb_find_perm(perm, ref_perm):
#     """Find the permutation that brings one list to the other. This is useful
#     for sorting by parties."""
#     res = np.zeros(perm.shape[0], dtype=uint16_)
#     for i in range(perm.shape[0]):
#         res[i] = nb_linsearch(perm, ref_perm[i])
#     return res

@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def mon_lexsorted(mon: np.ndarray, lexorder: np.ndarray) -> np.ndarray:
    """Sorts a monomial lexicographically."""
    mon_lexrepr = nb_mon_to_lexrepr(mon, lexorder)
    return mon[np.argsort(mon_lexrepr, kind='quicksort')]  # TODO Slow?, find a better way


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_op_lessthan_op(op1: np.array, op2: np.array, lexorder: np.ndarray) -> bool_:
    """Compares the lexicographic rank of two operators."""
    return nb_op_lexorder(op1, lexorder) < nb_op_lexorder(op2, lexorder)


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_op1_commuteswith_op2(op1: np.array, op2: np.array,
                            commutation_mat: np.ndarray,
                            lexorder: np.ndarray
                            ) -> bool_:
    """Compares the lexicographic rank of two operators. commutation_mat
    is a matrix whose (i, j) entry is 1 if the i-th operator commutes with
    the j-th. i, j are the lexicographic ranks of the operators.
    """
    return commutation_mat[nb_op_lexorder(op1, lexorder),
                           nb_op_lexorder(op2, lexorder)]


################################################################################
# FUNCTIONS WITH ARRAY OPERATIONS                                              #
################################################################################
@jit(nopython=True, cache=cache)
def A_lessthan_B(A: np.array, B: np.array) -> bool_:
    """Compare two letters/measurement operators lexicographically.

    Parameters
    ----------
    A : np.array
        Measurement operator encoded as a 1D array.
    B : np.array
        Measurement operator encoded as a 1D array.

    Returns
    -------
    bool
        Whether A is lexicographically smaller than B.
    """
    for i in range(A.shape[0]):
        if A[i] != B[i]:
            return A[i] < B[i]
    return True


@jit(nopython=True)
def nb_first_index(array: np.ndarray, item: float) -> int:
    """Find the first index of an item in an array.

    Parameters
    ----------
    array : numpy.ndarray
         The array to search.
    item : float
        The item to find.

    Returns
    -------
    int
        The index where the first item is found.

    Examples
    --------
    >>> array = numpy.array([1, 2, 3, 4, 5, 6])
    >>> nb_first_index(array, 5)
    4
    """
    for idx, val in enumerate(array):
        if abs(val - item) < 1e-10:
            return idx


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def reverse_mon(mon: np.ndarray) -> np.ndarray:
    """Output the monomial reversed, which means reverse the row of the 2d
    matrix representing the monomial. This represents the complex conjugate
    of the monomial, but we assume they are Hermitian.
    """

    return np.flipud(mon)


@jit(nopython=True)
def nb_unique(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find the unique elements in an array without sorting and in order of
    appearance.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to search.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        The unique values unsorted and their indices.

    Examples
    --------
    >>> nb_unique(np.array([1, 3, 3, 2, 2, 5, 4]))
    (array([1, 3, 2, 5, 4], dtype=int16), array([0, 1, 3, 5, 6], dtype=int16))
    """
    uniquevals = np.unique(arr)
    nr_uniquevals = uniquevals.shape[0]

    indices = np.zeros(nr_uniquevals).astype(int16_)
    for i in range(nr_uniquevals):
        indices[i] = nb_first_index(arr, uniquevals[i])
    indices.sort()

    uniquevals_unsorted = np.zeros(nr_uniquevals).astype(int16_)
    for i in range(nr_uniquevals):
        # Undo the sorting done by np.unique()
        uniquevals_unsorted[i] = arr[indices[i]]

    return uniquevals_unsorted, indices


################################################################################
# ABSTRACT OPERATIONS ON MONOMIALS                                             #
################################################################################
@jit(nopython=True, cache=cache)
def commuting(letter1: np.array, letter2: np.array, lexorder: np.array) -> bool_:
    """Determine if two letters/operators commute. Currently this only takes
    into account commutations coming of inflation

    Parameters
    ----------
    letter1 : np.array
        Tuple of integers representing an operator.
    letter2 : np.array
        Tuple of integers representing an operator.

    lexorder : np.array
        Specifies the order of the parties in the lexicographic order.
        Warning: Parties must be indexed starting from 1, and not 0.

    Returns
    -------
    bool
        Whether the letters commute or not.

    Examples
    --------
    A^11_00 commutes with A^22_00
    >>> commuting(np.array([1, 1, 1, 0, 0]), np.array([1, 2, 2, 0, 0]))
    True

    A^11_00 does not commute with A^12_00 since they overlap on source 1.
    >>> commuting(np.array([1, 1, 1, 0, 0]), np.array([1, 1, 2, 0, 0]))
    False
    """
    if letter1[0] != letter2[0]:
        return True
    if np.array_equal(letter1[1:-1], letter2[1:-1]):
        return True

    inf1, inf2 = letter1[1:-2], letter2[1:-2]
    inf1, inf2 = inf1[np.nonzero(inf1)], inf2[np.nonzero(inf2)]

    # If at least one in inf1-inf2 is 0, then there is one source in common
    # and therefore the letters don't commute.
    # If all are 0, then this case is covered in the first conditional,
    # they commute regardless of the value of the output
    return True if np.all(inf1-inf2) else False


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def mon_sorted_by_parties(mon: np.ndarray, lexorder: np.array) -> np.ndarray:
        """Sort by parties the monomial, i.e., sort by the first column in
        the 2d representation of the monomial.
        """
    PARTY_ORDER, _ = nb_unique(lexorder[:, 0])
    mon_sorted = np.zeros_like(mon)
    i_old = 0
    i_new = 0
    for i in range(PARTY_ORDER.shape[0]):
        pblock = mon[mon[:, 0] == PARTY_ORDER[i]]
        i_new += pblock.shape[0]
        mon_sorted[i_old:i_new] = pblock
        i_old = i_new
    return mon_sorted


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def mon_times_mon(mon1: np.ndarray,
                  mon2: np.ndarray
                  ) -> np.ndarray:
    """Return the product of two monomials, which is a concatenation.
    Warning: it does not simplify the resulting monomial!

    Parameters
    ----------
    mon1 : np.ndarray
        First monomial as a numpy array.
    mon2 : np.ndarray
        Second monomial as a numpy array.

    Returns
    -------
    np.ndarray
        Product of the two monomials.
    """

    return np.concatenate((mon1, mon2))

@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def mon_equal_mon(mon1: np.ndarray, mon2: np.ndarray) -> bool_:
    if mon1.shape != mon2.shape:
        return False
    else:
        #return np.array_equal(mon1, mon2)  # slower
        for i in range(mon1.shape[0]):
            if not nb_op_eq_op(mon1[i], mon2[i]):
                return False
        return True


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def dot_mon(mon1: np.ndarray,
            mon2: np.ndarray,
            lexorder: np.array
            ) -> np.ndarray:
    """Returns ((mon1)^dagger)*mon2.

    For hermitian operators this is the same as reversed(mon1)*mon2.
    Since all parties commute, the output is ordered by parties. We do
    not assume any other commutation rules.

    Parameters
    ----------
    mon1 : numpy.ndarray
        Monomial as a numpy array.
    mon2 : numpy.ndarray
        Monomial as a numpy array.

    Returns
    -------
    numpy.ndarray
        The monomial ordered by parties.

    Examples
    --------
    >>> mon1 = np.array([[1,2,3],[4,5,6]])
    >>> mon2 = np.array([[7,8,9]])
    >>> dot_mon(mon1, mon2)
    np.array([[4,5,6],[1,2,3],[7,8,9]])
    """
    if mon1.size <= 1:
        return mon2
    if mon2.size <= 1:
        return mon_sorted_by_parties(reverse_mon(mon1), lexorder)
    return mon_sorted_by_parties(np.concatenate((reverse_mon(mon1), mon2)),
                                 lexorder)

# @guvectorize(["void(int8[:, :], int8[:, :], int8[:, :])",
#               "void(int32[:, :], int32[:, :], int32[:, :])",
#               "void(int64[:, :], int64[:, :], int64[:, :])"],
#              '(n, m), (n, m) -> (n, m)',
#              nopython=True, cache=False)
# def dot_mon(mon1: np.ndarray,
#             mon2: np.ndarray,
#             res: np.ndarray
#             ):
#     res = _dot_mon(mon1, mon2)


# @jit(nopython=nopython, cache=cache, forceobj=not nopython) Cannot handle lexsort.
def dot_mon_commuting(mon1: np.ndarray,
                      mon2: np.ndarray,
                      lexorder: np.ndarray,
                      ) -> np.ndarray:
    """A faster implementation of `dot_mon` that assumes that all
    operators commute. This implies we order everything lexicographically.

    Parameters
    ----------
    mon1 : numpy.ndarray
        Monomial as a numpy array.
    mon2 : numpy.ndarray
        Monomial as a numpy array.

    Returns
    -------
    numpy.ndarray
        The dot product :math:`M_1^\dagger M_2` with the assumption that
        everything commutes with everything.
    """
    if mon1.size <= 1:
        if mon2.size <= 1:
            return mon2
        return mon_lexsorted(mon2, lexorder)
    if mon2.size <= 1:
        return mon_lexsorted(mon1, lexorder)

    return mon_lexsorted(np.concatenate((mon1, mon2)), lexorder)

@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def remove_projector_squares(mon: np.ndarray) -> np.ndarray:
    """Simplify the monomial by removing the squares. This is because we
    assume projectors, P^2=P.

    Parameters
    ----------
    mon : numpy.ndarray
        Input monomial as 2d array.

    Returns
    -------
    bool
        Whether the monomial is zero.
    """

    to_keep = np.ones(mon.shape[0], dtype=bool_)
    for i in range(1, mon.shape[0]):
        if nb_op_eq_op(mon[i], mon[i-1]):
            to_keep[i] = False
    return mon[to_keep]


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def mon_is_zero(mon: np.ndarray) -> bool_:
    """Function which checks if there is a product of two orthogonal projectors,
    and returns True if so."""
    for i in range(1, mon.shape[0]):
        if mon[i, -1] != mon[i-1, -1] and nb_op_eq_op(mon[i, :-1], mon[i-1, :-1]):
            return True
    return False


@jit(nopython=True, cache=cache)
def mon_sorted_by_parties(mon: np.ndarray) -> np.ndarray:
    """Sort the monomial by parties, i.e., sort by the first column in
    the 2d representation of the monomial.

    Parameters
    ----------
    mon : numpy.ndarray
        Input monomial as 2d array.

    Returns
    -------
    numpy.ndarray
        The sorted monomial.

    Examples
    --------
    >>> mon_sorted_by_parties(np.array([[3,...],[1,...],[4,...]]))
    np.array([[1,...],[3,...],[4,...]])
    """

    if len(monomial_numbers) == 0:
        return '1'
    else:
        return '*'.join(['_'.join([parties_names[letter[0] - 1]] +
                                  [str(i) for i in letter[1:]])
                         for letter in np.asarray(monomial_numbers).tolist()])


# def to_tuples(monomial: np.ndarray):
#     return tuple(tuple(vec) for vec in monomial.tolist())

def to_hashable(monomial: np.ndarray):
    return monomial.tobytes()


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_commuting(letter1: np.array,
              letter2: np.array
              ) -> bool_:
    """Determine if two letters/operators commute.

    TODO accept arbitrary commutation rules.
    Currently this only takes into accounts commutation coming of inflation

    Parameters
    ----------
    mon : numpy.ndarray
        Input monomial as 2d array.

    Returns
    -------
    numpy.ndarray
        The reversed monomial.

    Examples
    --------
    A^11_00 commutes with A^22_00
    >>> nb_commuting(np.array([1, 1, 1, 0, 0]), np.array([1, 2, 2, 0, 0]))
    True

    A^11_00 does not commute with A^12_00 because they overlap on source 1.
    >>> nb_commuting(np.array([1, 1, 1, 0, 0]), np.array([1, 1, 2, 0, 0]))
    False
    """
    return mon[np.arange(mon.shape[0])[::-1]]

def notcomm_from_lexorder(lexorder: np.ndarray)  -> np.ndarray:
    notcomm = np.zeros((lexorder.shape[0], lexorder.shape[0]), dtype=bool)
    for i in range(lexorder.shape[0]):
        for j in range(i + 1, lexorder.shape[0]):
            notcomm[i, j] = int(not nb_commuting(lexorder[i],
                                                 lexorder[j]))
    notcomm = notcomm + notcomm.T
    return notcomm

################################################################################
# OPERATIONS ON MONOMIALS RELATED TO INFLATION                                 #
################################################################################
@jit(nopython=True)
def apply_source_swap_monomial(monomial: np.ndarray,
                               source: int,
                               copy1: int,
                               copy2: int
                               ) -> np.ndarray:
    """Applies a swap of two sources to a monomial.

    Parameters
    ----------
    monomial : numpy.ndarray
        2d array representation of a monomial.
    source : int
         Integer in values [0, ..., nr_sources]
    copy1 : int
        Represents the copy of the source that swaps with copy2
    copy2 : int
        Represents the copy of the source that swaps with copy1

    Returns
    -------
    numpy.ndarray
         The new monomial with swapped sources.

    Examples
    --------
    >>> monomial = np.array([[1, 2, 3, 0, 0, 0]])
    >>> apply_source_swap_monomial(np.array([[1, 0, 2, 1, 0, 0],
                                             [2, 1, 3, 0, 0, 0]]),
                                             1,  # source
                                             2,  # copy1
                                             3)  # copy2
    array([[1, 0, 3, 1, 0, 0],
           [2, 1, 2, 0, 0, 0]])
    """
    new_factors = monomial.copy()
    for i in range(new_factors.shape[0]):
        copy = new_factors[i, 1 + source]
        if copy > 0:
            if copy == copy1:
                new_factors[i, 1 + source] = copy2
            elif copy == copy2:
                new_factors[i, 1 + source] = copy1
    return new_factors


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def mon_lessthan_mon(mon1: np.ndarray,
                     mon2: np.ndarray,
                     lexorder: np.ndarray,
                     ) -> bool_:
    """Compares two monomials and returns True if mon1 < mon2 in lexicographic
    order.

    Parameters
    ----------
    mon1 : numpy.ndarray
        Monomial encoded as a 2d array where each row is an operator.
    mon2 : numpy.ndarray
        Monomial encoded as a 2d array where each row is an operator.
    lexorder : numpy.array
        The 2d array encoding the default lexicographical order.

    Returns
    -------
    bool
        Whether mon1 < mon2.
    """
    mon1_lexrank = nb_mon_to_lexrepr(mon1, lexorder)
    mon2_lexrank = nb_mon_to_lexrepr(mon2, lexorder)

    #return A_lessthan_B(mon1.ravel(), mon2.ravel())
    return A_lessthan_B(mon1_lexrank, mon2_lexrank)


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_apply_substitutions(mon_in: np.ndarray, notcomm: np.ndarray, lexorder: np.ndarray) -> np.ndarray:
    """Apply substitutions to a monomial.

    Currently it only supports commutations arising from operators having
    completely different support. It goes in a loop applying the substitutions
    until it reaches a fixed point, if it finds two letters that commute and
    are not in lexicographic ordering. This function does a single loop from
    the first row to the last applying all substitutions along the way and
    then it returns.

    Parameters
    ----------
    mon_in : numpy.ndarray
        Input monomial as a 2d array.

    Returns
    -------
    numpy.ndarray
        The input monomial simplified after substitutions.
    """
    if mon_in.shape[0] == 1:
        return mon_in
    mon = mon_in.copy()
    #mon = nb_mon_to_lexrepr(mon_in, lexorder)
    for i in range(1, mon.shape[0]):
        if not A_lessthan_B(mon[i-1], mon[i]):
            if nb_commuting(mon[i-1], mon[i]):
                mon[i-1, :], mon[i, :] = mon[i, :].copy(), mon[i-1, :].copy()
    return mon
    #return lexorder[mon]


#@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def to_canonical(mon: np.ndarray, notcomm: np.ndarray, lexorder: np.ndarray
                 ) -> np.ndarray:
    """Apply substitutions to a monomial until it stops changing.

    Parameters
    ----------
    mon : numpy.ndarray
        Input monomial as a 2d array.

    Returns
    -------
    numpy.ndarray
        The monomial in canonical form with respect to some lexicographic
        ordering.
    """
    if mon.shape[0] <= 1:
        return mon
    else:
        mon_lexorder = nb_mon_to_lexrepr(mon, lexorder)
        mon = nb_to_canonical_lexinput(mon_lexorder, notcomm)
        mon = lexorder[mon]
        mon = remove_projector_squares(mon)

        return mon


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_to_canonical_lexinput(mon_lexorder: np.array, notcomm: np.ndarray
                                ) -> np.array:
    if mon_lexorder.shape[0] <= 1:
        return mon_lexorder

    # Take only the rows and columns of notcomm that appear in the monomial,
    # in the correct order.

    sub_notcomm = notcomm[mon_lexorder, :][:, mon_lexorder]  # TODO take this outside
    # comm_paths_toleft = np.zeros((mon_lexorder.shape[0], mon_lexorder.shape[0]), dtype=int)

    # idx = nb_linsearch(sub_notcomm[0], 1)
    # if idx == 1: # If the first operator cannot be moved at all
    #     m1 = np.array([mon_lexorder[0]])
    #     m2 = to_canonical(mon_lexorder[1:], notcomm)
    #     print(1, m1, m2)
    #     return np.concatenate((m1, m2))
    # else:
    minimo = mon_lexorder[0]
    minimo_idx = 0
    for op in range(1, mon_lexorder.shape[0]):
        #print(sub_notcomm[op][:op])
        #idx = nb_linsearch(sub_notcomm[op][:op], np.ones(1, dtype=np.int32)[0])
        where = np.where(sub_notcomm[op, :op] == 1)[0]
        if where.size < 1: # TODO make nb_linsearch work, its faster
        #if idx < 0: # means no collider was found
            if mon_lexorder[op] < minimo:
                minimo_idx = op
                minimo = mon_lexorder[op]
    if minimo <= mon_lexorder[0]:
        m1 = np.array([mon_lexorder[minimo_idx]])
        m2 = np.concatenate((mon_lexorder[:minimo_idx], mon_lexorder[minimo_idx+1:]))
        return np.concatenate((m1, nb_to_canonical_lexinput(m2, notcomm)))
    else:
        m1 = np.array([mon_lexorder[0]])
        m2 = mon_lexorder[1:]
        return np.concatenate((m1, nb_to_canonical_lexinput(m2, notcomm)))


def to_name(monomial: np.ndarray, names: List[str]) -> str:
    """Convert the 2d array representation of a monomial to a string.

    Parameters
    ----------
    monomial : numpy.ndarray
        Monomial in matrix format.
    names : List[str]
        List of party names.

    Returns
    -------
    str
        The string representation of the monomial.

    Examples
    --------
    >>> to_name([[1 1,0,3], [4,1,2,6], [2,3,3,4]], ['a','bb','x','Z'])
    'a_1_0_3*Z_1_2_6*bb_3_3_4'
    """
    if mon_equal_mon(monomial, np.array([0])):
        return '1'

    # It is faster to convert to list of lists than to loop through numpy arrays
    monomial = monomial.tolist()
    return '*'.join(['_'.join([names[letter[0]-1]]+[str(i) for i in letter[1:]])
                     for letter in monomial])


################################################################################
# OPERATIONS ON MOMENT MATRICES                                                #
################################################################################
def calculate_momentmatrix(cols: List,
                           notcomm,
                           lexorder,
                           verbose: int = 0,
                           commuting=False,
                           dtype: np.dtype = np.uint16
                           ) -> Tuple[np.ndarray, Dict]:
    r"""Calculate the moment matrix. The function takes as input the generating
    set :math:`\{M_i\}_i` encoded as a list of monomials. Each monomial is a
    matrix where each row is an operator and the columns specify the operator
    labels/indices. The moment matrix is the inner product between all possible
    pairs of elements from the generating set. The program outputs the moment
    matrix as a 2d array. Entry :math:`(i,j)` of the moment matrix stores the
    index of the monomial that represents the result of the expectation value
    :math:`\text{Tr}(\rho\cdot M_i^\dagger M_j)` for an unknown quantum state
    :math:`\rho` after applying the substitutions. The program returns the
    moment matrix and the dictionary mapping each monomial in string
    representation to its integer representation.

    Parameters
    ----------
    cols : List
        List of numpy.ndarray representing the generating set.
    names : numpy.ndarray
        The string names of each party.
    commuting: bool, optional
        Whether the variables in the problem commute or not. By default
        ``False``.
    verbose : int, optional
        How much information to print. By default ``0``.

    Returns
    -------
    Tuple[numpy.ndarray, Dict]
        The moment matrix :math:`\Gamma`, where each entry :math:`(i,j)` stores
        the integer representation of a monomial. The Dict is a mapping from
        string representation to integer representation.
    """
    # Choose dot_mon depending on commutation properties
    if commuting:
        dot_fn = dot_mon_commuting
    else:
        dot_fn = dot_mon
    nrcols = len(cols)
    canonical_mon_to_idx_dict = dict()
    momentmatrix = dok_matrix((nrcols, nrcols), dtype=np.uint32)
    varidx = 1  # We start from 1 because 0 is reserved for 0
    for i in tqdm(range(nrcols),
                  disable=not verbose,
                  desc="Calculating moment matrix    "):
        for j in range(i, nrcols):
            mon1, mon2 = cols[i], cols[j]
            if not commuting:
                mon_v1 = dot_mon(mon1, mon2, lexorder)
            else:
                mon_v1 = dot_mon_commuting(mon1, mon2, lexorder)
            if mon_is_zero(mon_v1):
                # If sparse, we don't need this, but for readibility...
                momentmatrix[i, j] = 0
            else:
                if not commuting:
                    mon_v1 = to_canonical(mon_v1, notcomm, lexorder).astype(dtype)
                    mon_v2 = to_canonical(dot_mon(mon2, mon1, lexorder), notcomm, lexorder).astype(dtype)
                    mon_hash = min(mon_v1.tobytes(), mon_v2.tobytes())

                else:
                    mon = remove_projector_squares(mon_v1).astype(dtype)
                    # mon_as_tuples = to_tuples(mon)
                    mon_hash = mon.tobytes()
                try:
                    known_varidx = canonical_mon_to_idx_dict[mon_hash]
                    momentmatrix[i, j] = known_varidx
                    momentmatrix[j, i] = known_varidx
                except KeyError:
                    canonical_mon_to_idx_dict[mon_hash] = varidx
                    momentmatrix[i, j] = varidx
                    momentmatrix[j, i] = varidx
                    varidx += 1
    return momentmatrix.todense(), canonical_mon_to_idx_dict


def factorize_monomial(raw_monomial: np.ndarray,
                       canonical_order=True
                       ) -> Tuple[np.ndarray]:
    """This function splits a moment/expectation value into products of
    moments according to the support of the operators within the moment.

    The moment is encoded as a 2d array where each row is an operator.
    If monomial=A*B*C*B then row 1 is A, row 2 is B, row 3 is C and row 4 is B.
    In each row, the columns encode the following information:

    First column:       The party index, *starting from 1*.
                        (1 for A, 2 for B, etc.)
    Last two columns:   The input x, starting from zero and then the
                        output a, starting from zero.
    In between:         This encodes the support of the operator. There
                        are as many columns as sources/quantum states.
                        Column j represents source j-1 (-1 because the 1st
                        col is the party). If the value is 0, then this
                        operator does not measure this source. If the value
                        is for e.g. 2, then this operator is acting on
                        copy 2 of source j-1.

    The output is a tuple of ndarrays where each array represents another
    monomial s.t. their product is equal to the original monomial.

    Parameters
    ----------
    monomial : np.ndarray
        Monomial encoded as a 2d array where each row is an operator.

    Returns
    -------
    Tuple[np.ndarray]
        A tuple of ndarrays, where each array represents an atomic monomial factor.

    Examples
    --------
    >>> monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [2, 1, 0, 2, 0, 0],
                             [1, 0, 3, 3, 0, 0],
                             [3, 3, 5, 0, 0, 0],
                             [3, 1, 4, 0, 0, 0],
                             [3, 6, 6, 0, 0, 0],
                             [3, 4, 5, 0, 0, 0]])
    >>> factorised = factorize_monomial(monomial)
    [array([[1, 0, 1, 1, 0, 0]]),
     array([[1, 0, 3, 3, 0, 0]]),
     array([[2, 1, 0, 2, 0, 0],
            [3, 1, 4, 0, 0, 0]]),
     array([[3, 3, 5, 0, 0, 0],
            [3, 4, 5, 0, 0, 0]]),
     array([[3, 6, 6, 0, 0, 0]])]

    """
    if len(raw_monomial) == 0:
        return [raw_monomial]

    monomial = np.asarray(raw_monomial, dtype=np.uint16)
    components_indices = np.zeros((len(monomial), 2), dtype=np.uint16)
    # Add labels to see if the components have been used
    components_indices[:, 0] = np.arange(0, len(monomial), 1)

    inflation_indices = monomial[:, 1:-2]
    disconnected_components = []

    idx = 0
    while idx < len(monomial):
        component = []
        if components_indices[idx, 1] == 0:
            component.append(idx)
            components_indices[idx, 1] = 1
            jdx = 0
            # Iterate over all components that are connected
            while jdx < len(component):
                nonzero_sources = np.nonzero(
                    inflation_indices[component[jdx]])[0]
                for source in nonzero_sources:
                    overlapping = inflation_indices[:,
                           source] == inflation_indices[component[jdx], source]
                    # Add the components that overlap to the lookup list
                    component += components_indices[overlapping &
                                (components_indices[:, 1] == 0)][:, 0].tolist()
                    # Specify that the components that overlap have been used
                    components_indices[overlapping, 1] = 1
                jdx += 1
        if len(component) > 0:
            disconnected_components.append(component)
        idx += 1

    disconnected_components = tuple(
        monomial[sorted(component)] for component in disconnected_components)

    # We would like to have a canonical ordering of the factors.
    if canonical_order:
        disconnected_components = tuple(sorted(disconnected_components, key=to_hashable))

    return disconnected_components

if __name__ == '__main__':
    import numpy as np
    import timeit
    lexorder = np.array([[1, 1, 1, 0, 0, 0],
                         [1, 1, 2, 0, 0, 0],
                         [1, 2, 1, 0, 0, 0],
                         [1, 2, 2, 0, 0, 0],
                         [1, 3, 2, 0, 0, 0],
                         [2, 1, 0, 1, 0, 0],
                         [2, 1, 0, 2, 0, 0],
                         [2, 2, 0, 1, 0, 0],
                         [2, 2, 0, 2, 0, 0],
                         [3, 0, 1, 1, 0, 0],
                         [3, 0, 1, 2, 0, 0],
                         [3, 0, 2, 1, 0, 0],
                         [3, 0, 2, 2, 0, 0]])

    notcomm = np.zeros((lexorder.shape[0], lexorder.shape[0]), dtype=int)
    notcomm[0, 1] = 1; notcomm[0, 2] = 1;
    notcomm[1, 3] = 1
    notcomm[1, 4] = 1
    notcomm[2, 3] = 1
    notcomm[3, 4] = 1
    notcomm[5, 6] = 1; notcomm[5, 7] = 1;
    notcomm[6, 8] = 1
    notcomm[7, 8] = 1
    notcomm[8, 10] = 1; notcomm[9, 11] = 1;
    notcomm[10, 12] = 1
    notcomm[11, 12] = 1
    notcomm = notcomm + notcomm.T

    np.random.seed(132)

    mon_lexorder = np.random.permutation(lexorder.shape[0])[:7]



    @jit(nopython=True)
    def nb_to_canonical_lexinput(mon_lexorder: np.array, notcomm: np.ndarray
                                 ) -> np.array:
        if mon_lexorder.shape[0] <= 1:
            return mon_lexorder

        # Take only the rows and columns of notcomm that appear in the monomial,
        # in the correct order.

        sub_notcomm = notcomm[mon_lexorder, :][:, mon_lexorder]  # TODO take this outside
        # comm_paths_toleft = np.zeros((mon_lexorder.shape[0], mon_lexorder.shape[0]), dtype=int)

        # idx = nb_linsearch(sub_notcomm[0], 1)
        # if idx == 1: # If the first operator cannot be moved at all
        #     m1 = np.array([mon_lexorder[0]])
        #     m2 = to_canonical(mon_lexorder[1:], notcomm)
        #     print(1, m1, m2)
        #     return np.concatenate((m1, m2))
        # else:
        minimo = mon_lexorder[0]
        minimo_idx = 0
        for op in range(1, mon_lexorder.shape[0]):
            #print(sub_notcomm[op][:op])
            #idx = nb_linsearch(sub_notcomm[op][:op], np.ones(1, dtype=np.int32)[0])
            where = np.where(sub_notcomm[op, :op] == 1)[0]
            if where.size < 1: # TODO make nb_linsearch work, its faster
            #if idx < 0: # means no collider was found
                if mon_lexorder[op] < minimo:
                    minimo_idx = op
                    minimo = mon_lexorder[op]
        if minimo <= mon_lexorder[0]:
            m1 = np.array([mon_lexorder[minimo_idx]])
            m2 = np.concatenate((mon_lexorder[:minimo_idx], mon_lexorder[minimo_idx+1:]))
            return np.concatenate((m1, nb_to_canonical_lexinput(m2, notcomm)))
        else:
            m1 = np.array([mon_lexorder[0]])
            m2 = mon_lexorder[1:]
            return np.concatenate((m1, nb_to_canonical_lexinput(m2, notcomm)))
