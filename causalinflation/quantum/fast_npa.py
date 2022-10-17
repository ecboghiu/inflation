"""
This file contains helper functions to manipulate monomials and generate moment
matrices. The functions in this file can be accelerated by JIT compilation in
numba.
@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""
from typing import List, Dict, Tuple, Union
import numpy as np
from scipy.sparse import dok_matrix

try:
    from numba import jit
    from numba.types import bool_, void
    from numba.types import uint16 as uint16_
    from numba.types import int16 as int16_
    from numba.types import int64 as int64_
    from numba.typed import Dict as nb_Dict
except ImportError:
    def jit(*args, **kwargs):
        return lambda f: f
    bool_   = bool
    uint16_ = np.uint16
    int16_  = np.int16
    int64_  = np.int
    nb_Dict = dict
    void    = None

try:
    from tqdm import tqdm
except ImportError:
    from ..utils import blank_tqdm as tqdm

cache    = True
nopython = True
if not nopython:
    bool_   = bool
    uint16_ = np.uint16
    int64_  = np.int
    nb_Dict = dict


###############################################################################
# ABSTRACT OPERATIONS ON MONOMIALS                                            #
###############################################################################
@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def reverse_mon(mon: np.ndarray) -> np.ndarray:
    """Return the reversed monomial.

    This represents the Hermitian conjugate of the monomial, assuming that
    each operator is Hermitian.

    Parameters
    ----------
    mon : np.ndarray
        Monomial as a matrix with rows as integer arrays representing operators.

    Returns
    -------
    np.ndarray
        Reversed monomial.
    """
    return np.flipud(mon)


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def remove_projector_squares(mon: np.ndarray) -> np.ndarray:
    """Simplify the monomial by removing operator powers. This is because we
    assume projectors, P**2=P. This corresponds to removing duplicates of rows
    which are adjacent.

    Parameters
    ----------
    mon : numpy.ndarray
        Monomial as a matrix with rows as integer arrays representing operators.

    Returns
    -------
    bool
        Monomial with powers of operators removed.
    """
    to_keep = np.ones(mon.shape[0], dtype=bool_)
    for i in range(1, mon.shape[0]):
        if np.array_equal(mon[i], mon[i - 1]):
            to_keep[i] = False
    return mon[to_keep]


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def mon_is_zero(mon: np.ndarray) -> bool_:
    """Function which checks if
    1) there is a product of two orthogonal projectors,
    2) or the monomial is equal to the canonical zero monomial
    and returns True if so.

    Parameters
    ----------
    mon : np.ndarray
        Monomial as a matrix with rows as integer arrays representing operators.

    Returns
    -------
    bool
        Whether the monomial evaluates to zero.
    """
    if len(mon) >= 1 and not np.any(mon.ravel()):
        return True
    for i in range(1, mon.shape[0]):
        if ((mon[i, -1] != mon[i - 1, -1])
                and np.array_equal(mon[i, :-1], mon[i - 1, :-1])):
            return True
    return False


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_op_lexorder(operator: np.ndarray,
                   lexorder: np.ndarray) -> int:
    """Map each operator to a unique integer for hashing and lexicographic
    ordering comparison purposes.

    Warning: if `operator` is not found in `lexorder`, the function will
    not return any value.

    Parameters
    ----------
    operator : np.ndarray
        Array of integers representing the operator.
    lexorder : np.ndarray
        Matrix with rows as operators where the index of the row gives
        the lexicographic ordering.

    Returns
    -------
    int
        The index of the operator in the lexorder matrix.
    """
    for i in range(lexorder.shape[0]):
        if np.array_equal(lexorder[i, :], operator):
            return i


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_mon_to_lexrepr(mon: np.ndarray,
                      lexorder: np.ndarray) -> np.array:
    """Convert a monomial to its lexicographic representation, as an
    array of integers representing the lex rank of each operator.

    Parameters
    ----------
    mon : np.ndarray
        Monomial as a matrix with rows as integer arrays representing operators.
    lexorder : np.ndarray
        Matrix with rows as operators where the index of the row gives
        the lexicographic order of the operator.

    Returns
    -------
    np.array
        Monomial as array of integers, where each integer is the hash
        of the corresponding operator.
    """
    lex = np.zeros_like(mon[:, 0])
    for i in range(mon.shape[0]):
        lex[i] = nb_op_lexorder(mon[i], lexorder)
    return lex


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def mon_lexsorted(mon: np.ndarray,
                  lexorder: np.ndarray) -> np.ndarray:
    """Sorts a monomial lexicographically.

    Parameters
    ----------
    mon : np.ndarray
        Monomial as a matrix with rows as integer arrays representing operators.
    lexorder : np.ndarray
        Matrix with rows as operators where the index of the row gives
        the lexicographic order of the operator.

    Returns
    -------
    np.ndarray
        Sorted monomial.
    """
    mon_lexrepr = nb_mon_to_lexrepr(mon, lexorder)
    return mon[np.argsort(mon_lexrepr, kind="quicksort")]


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def dot_mon(mon1: np.ndarray,
            mon2: np.ndarray) -> np.ndarray:
    """Returns ((mon1)^dagger)*mon2.

    For hermitian operators this is the same as reversed(mon1)*mon2.
    Since all parties commute, the output is ordered by parties. We do
    not assume any other commutation rules.

    Parameters
    ----------
    mon1 : numpy.ndarray
        Monomial as a matrix with rows as integer arrays representing operators.
    mon2 : numpy.ndarray
        Monomial as a matrix with rows as integer arrays representing operators.

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
    return np.concatenate((reverse_mon(mon1), mon2))


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def to_name(monomial: Union[np.ndarray, List[List[int]]],
            names: List) -> str:
    """Convert the 2d array representation of a monomial to a string.

    Parameters
    ----------
    monomial : numpy.ndarray
        Monomial as a matrix with each row being an integer array representing
        an operators.
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
    if len(monomial) == 0:
        return "1"

    monomial = monomial.tolist()
    return "*".join(["_".join([names[letter[0] - 1]]
                              + [str(i) for i in letter[1:]])
                     for letter in monomial])


###############################################################################
# OPERATIONS ON MONOMIALS RELATED TO INFLATION                                #
###############################################################################
@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_commuting(operator1: np.ndarray,
                 operator2: np.ndarray) -> bool_:
    """Determine if two operators commute. Currently, this only takes
    into account commutation coming from inflation and settings.

    Parameters
    ----------
    operator1 : numpy.ndarray
        Operator as an array of integers.
    operator2 : numpy.ndarray
        Operator as an array of integers.

    Returns
    -------
    bool
        True if they commute, False if they do not.

    Examples
    --------
    A^11_00 commutes with A^22_00
    >>> nb_commuting(np.array([1, 1, 1, 0, 0]), np.array([1, 2, 2, 0, 0]))
    True

    A^11_00 does not commute with A^12_00 because they overlap on source 1.
    >>> nb_commuting(np.array([1, 1, 1, 0, 0]), np.array([1, 1, 2, 0, 0]))
    False
    """
    if operator1[0] != operator2[0]:  # Different parties
        return True
    elif np.array_equal(operator1[1:-1], operator2[1:-1]):  # = sources & settings
        return True
    else:
        inf1, inf2 = operator1[1:-2], operator2[1:-2]  # Compare just the sources
        inf1, inf2 = inf1[np.flatnonzero(inf1)], inf2[np.flatnonzero(inf2)]
        # If at least one in inf1-inf2 is 0, then there is one source in common
        # and therefore the letters don't commute.
        return np.all(inf1 - inf2)


def commutation_matrix(lexorder: np.ndarray, commuting=False) -> np.ndarray:
    """Helper function that builds a matrix encoding which operators commute
    according to the function `nb_commuting`. Rows and columns are indexed by
    operators, and each element is a 0 if the operators in the row and in the
    column commute or 1 if they do not.

    Parameters
    ----------
    lexorder : numpy.ndarray
        Matrix with rows as operators where the index of the row gives
        the lexicographic order of the operator.

    Returns
    -------
    numpy.ndarray
        Matrix whose entry :math:`(i,j)` has value 1 if the operators with
        lexicographic ordering :math:`i` and :math:`j` do not commute, and
        value 0 if they commute.
    """
    notcomm = np.zeros((lexorder.shape[0], lexorder.shape[0]), dtype=bool)
    if not commuting:
        for i in range(lexorder.shape[0]):
            for j in range(i + 1, lexorder.shape[0]):
                notcomm[i, j] = not nb_commuting(lexorder[i], lexorder[j])
        notcomm = notcomm + notcomm.T
    return notcomm


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def all_commuting_test(mon: np.ndarray,
                       lexorder: np.ndarray,
                       notcomm: np.ndarray) -> bool:
    if len(mon) <= 1:
        return True
    mon_lexorder = nb_mon_to_lexrepr(mon, lexorder)
    sub_notcomm = notcomm[mon_lexorder, :][:, mon_lexorder]
    return not sub_notcomm.any()


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_to_canonical_lexinput(mon_lexorder: np.ndarray,
                             notcomm: np.ndarray) -> np.ndarray:
    """Brings a monomial to canonical form with respect to commutations.

    This function works recursively. Assume
    `mon_lexorder=np.array([op0, op1, op2, op3, op4])`
    where the operators are encoded as integers representing their lexicographic
    ordering.

    First it checks whether there is any operator `opX` with a "commuting path"
    from that operator to `op0`. A commuting path from `opX` is a sequence of
    operators from `op0` to `opX` such that all of them commute. Consider
    the set of all operators that have a commuting path to `op0`. We take the
    smallest operator in this set in lexicographic ordering and place it in
    the first position, and displace `op0` to the second position. If `op0`
    cannot be displaced, we do not move it. Then we call
    `nb_to_canonical_lexinput` on the remaining 4 operators. If the smallest
    operator is `op3` in the above example, then we return:

    `[op0, op1, op2, op3, op4]` -> `[op3, nb_to_canonical_lexinput([op0, op1, op2, op4])]`

    This procedure is done recursively until it stops.

    Parameters
    ----------
    mon_lexorder : numpy.ndarray
        Monomial as an array of integers, where each integer represents
        the lexicographic order of an operator.
    notcomm : numpy.ndarray
        Matrix of commutation relations, given in the format specified by
        `causalinflation.quantum.fast_npa.commutation_matrix`.

    Returns
    -------
    np.ndarray
        Monomial in canonical form with respect to commutations.
    """
    if mon_lexorder.shape[0] <= 1:
        return mon_lexorder

    # Take only the rows and columns of notcomm that appear in the monomial,
    # in the correct order.
    sub_notcomm = notcomm[mon_lexorder, :][:, mon_lexorder]
    if not sub_notcomm.any():
        return np.sort(mon_lexorder)
    if sub_notcomm.all():
        return mon_lexorder
    minimum = mon_lexorder[0]
    minimum_idx = 0
    for op in range(1, mon_lexorder.shape[0]):
        where = np.where(sub_notcomm[op, :op] == 1)[0]
        if where.size < 1:
            if mon_lexorder[op] < minimum:
                minimum_idx = op
                minimum = mon_lexorder[op]
    if minimum <= mon_lexorder[0]:
        m1 = np.array([mon_lexorder[minimum_idx]])
        m2 = np.concatenate((mon_lexorder[:minimum_idx],
                             mon_lexorder[minimum_idx + 1:]))
        return np.concatenate((m1, nb_to_canonical_lexinput(m2, notcomm)))
    else:
        m1 = np.array([mon_lexorder[0]])
        m2 = mon_lexorder[1:]
        return np.concatenate((m1, nb_to_canonical_lexinput(m2, notcomm)))


# to_canonical is not numba-ifiable, but must be stored here due to circular
# import issues
def to_canonical(mon: np.ndarray,
                 notcomm: np.ndarray,
                 lexorder: np.ndarray,
                 commuting=False,
                 hasty=False) -> np.ndarray:
    """Brings a monomial to canonical form with respect to commutations,
    and removes square projectors, and identifies orthogonality.

    Parameters
    ----------
    mon : numpy.ndarray
        Monomial as a matrix with rows as integer arrays representing operators.
    commuting : bool, optional
        Whether the variables in the problem commute or not. By default
        ``False``.
    hasty : bool, optional
        Whether to skip the removal of projector squares and the test to see if the monomial is equal to zero.

    Returns
    -------
    numpy.ndarray
        The monomial in canonical form with respect to some commutation
        relationships.
    """
    if mon.shape[0] <= 1:
        return mon
    else:
        mon = hasty_to_canonical(mon, notcomm, lexorder, commuting=commuting)
        if hasty:
            return mon
        else:
            mon = remove_projector_squares(mon)
            if mon_is_zero(mon):
                return 0*mon[:1]
            else:
                return mon


def hasty_to_canonical(mon: np.ndarray,
                       notcomm: np.ndarray,
                       lexorder: np.ndarray,
                       commuting=False) -> np.ndarray:
    """Brings a monomial to canonical form with respect to commutations, but does
    not check for squared projectors or orthogonality.

    Parameters
    ----------
    mon : numpy.ndarray
        Monomial as a matrix with rows as integer arrays representing operators.

    Returns
    -------
    numpy.ndarray
        The monomial in canonical form with respect to some commutation
        relationships.
    """
    if len(mon) <= 1:
        return mon
    else:
        if commuting:
            mon = mon_lexsorted(mon, lexorder)
            return mon
        else:
            mon_lexorder = nb_mon_to_lexrepr(mon, lexorder)
            mon = nb_to_canonical_lexinput(mon_lexorder, notcomm)
            mon = lexorder[mon]
            return mon


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def apply_source_perm(monomial: np.ndarray,
                      source: int,
                      permutation: np.ndarray) -> np.ndarray:
    """Apply a source swap to a monomial.

    Parameters
    ----------
    monomial : numpy.ndarray
        Input monomial in 2d array format.
    source : int
        The source that is being swapped.
    permutation : numpy.ndarray
        The permutation of the copies of the specified source.
        The format for the permutation here is to use indexing starting at one,
        so the permutation must be padded with a leading zero. The function
        ``causalinflation.quantum.general_tools.format_permutations`` converts
        a list of permutations to the necessary format.

    Returns
    -------
    numpy.ndarray
        Input monomial with the specified source swapped.
    """
    new_factors = monomial.copy()
    new_factors[:, 1 + source] = permutation[new_factors[:, 1 + source]]
    return new_factors


###############################################################################
# OPERATIONS ON MOMENT MATRICES                                               #
###############################################################################
def calculate_momentmatrix(cols: List,
                           notcomm: np.ndarray,
                           lexorder: np.ndarray,
                           verbose: int = 0,
                           commuting: bool = False,
                           dtype: object = np.uint16
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
    notcomm : numpy.ndarray
        Matrix of commutation relations, given in the format specified by
        `causalinflation.quantum.fast_npa.commutation_matrix`.
    lexorder : numpy.ndarray
        Matrix with rows as operators where the index of the row gives
        the lexicographic order of the operator.
    commuting : bool, optional
        Whether the variables in the problem commute or not. By default
        ``False``.
    verbose : int, optional
        How much information to print. By default ``0``.
    dtype: np.dtype, optional
        The dtype for constructing monomials when represented as numpy arrays.

    Returns
    -------
    Tuple[numpy.ndarray, Dict]
        The moment matrix :math:`\Gamma`, where each entry :math:`(i,j)` stores
        the integer representation of a monomial. The Dict is a mapping from
        string representation to integer representation.
    """
    nrcols = len(cols)
    canonical_mon_to_idx_dict = dict()
    momentmatrix = dok_matrix((nrcols, nrcols), dtype=np.uint32)
    varidx = 1  # We start from 1 because 0 is reserved for 0
    for i in tqdm(range(nrcols),
                  disable=not verbose,
                  desc="Calculating moment matrix"):
        for j in range(i, nrcols):
            mon1, mon2 = cols[i], cols[j]
            mon_v1 = to_canonical(dot_mon(mon1, mon2).astype(dtype),
                                  notcomm,
                                  lexorder,
                                  commuting=commuting)
            if not mon_is_zero(mon_v1):
                if not commuting:
                    mon_v2 = to_canonical(dot_mon(mon2, mon1).astype(dtype),
                                          notcomm,
                                          lexorder,
                                          commuting=commuting)
                    mon_hash = min(mon_v1.tobytes(), mon_v2.tobytes())
                else:
                    mon_hash = mon_v1.tobytes()
                try:
                    known_varidx = canonical_mon_to_idx_dict[mon_hash]
                    momentmatrix[i, j] = known_varidx
                    momentmatrix[j, i] = known_varidx
                except KeyError:
                    canonical_mon_to_idx_dict[mon_hash] = varidx
                    momentmatrix[i, j] = varidx
                    momentmatrix[j, i] = varidx
                    varidx += 1
    return momentmatrix.toarray(), canonical_mon_to_idx_dict
