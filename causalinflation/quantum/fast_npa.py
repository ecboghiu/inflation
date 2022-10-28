"""
This file contains helper functions to manipulate monomials and generate moment
matrices. The functions in this file can be accelerated by JIT compilation in
numba.
@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""
import numpy as np

from typing import List

try:
    from numba import jit
    from numba.types import bool_, void
    from numba.types import uint8 as uint8_
except ImportError:
    def jit(*args, **kwargs):
        return lambda f: f
    bool_  = bool
    uint8_ = np.uint8
    void   = None

cache    = True
nopython = True
if not nopython:
    bool_  = bool
    uint8_ = np.uint8


###############################################################################
# ABSTRACT OPERATIONS ON MONOMIALS                                            #
###############################################################################
@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def dot_mon(mon1: np.ndarray,
            mon2: np.ndarray) -> np.ndarray:
    """Returns ((mon1)^dagger)*mon2.

    For hermitian operators this is the same as reversed(mon1)*mon2. Since all
    parties commute, the output is ordered by parties. We do not assume any
    other commutation rules.

    Parameters
    ----------
    mon1 : numpy.ndarray
        Input monomial in 2d array format.
    mon2 : numpy.ndarray
        Input monomial in 2d array format.

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
def mon_is_zero(mon: np.ndarray) -> bool_:
    """Function which checks if a monomial is equivalent to the zero monomial.
    This is the case if there is a product of two orthogonal projectors, or if
    the monomial is equal to the canonical zero monomial.

    Parameters
    ----------
    mon : numpy.ndarray
        Input monomial in 2d array format.

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
def mon_lexsorted(mon: np.ndarray,
                  lexorder: np.ndarray) -> np.ndarray:
    """Sorts a monomial lexicographically.

    Parameters
    ----------
    mon : numpy.ndarray
        Input monomial in 2d array format.
    lexorder : numpy.ndarray
        Matrix with rows as operators where the index of the row gives
        the lexicographic order of the operator.

    Returns
    -------
    numpy.ndarray
        Sorted monomial.
    """
    mon_lexrepr = nb_mon_to_lexrepr(mon, lexorder)
    return mon[np.argsort(mon_lexrepr, kind="quicksort")]


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_lexorder_idx(operator: np.ndarray,
                    lexorder: np.ndarray) -> int:
    """Return the unique integer corresponding to the lexicographic ordering of
    the operator. If ``operator`` is not found in ``lexorder``, no value is
    returned.

    Parameters
    ----------
    operator : numpy.ndarray
        Array of integers representing the operator.
    lexorder : numpy.ndarray
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
        Monomial as array of integers, where each integer is the hash
        of the corresponding operator.
    """
    lex = np.zeros_like(mon[:, 0])
    for i in range(mon.shape[0]):
        lex[i] = nb_lexorder_idx(mon[i], lexorder)
    return lex


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def remove_projector_squares(mon: np.ndarray) -> np.ndarray:
    """Simplify the monomial by removing operator powers. This is because we
    assume projectors, P**2=P. This corresponds to removing duplicates of rows
    which are adjacent.

    Parameters
    ----------
    mon : numpy.ndarray
        Input monomial in 2d array format.

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
def reverse_mon(mon: np.ndarray) -> np.ndarray:
    """Return the reversed monomial.

    This represents the Hermitian conjugate of the monomial, assuming that
    each operator is Hermitian.

    Parameters
    ----------
    mon : numpy.ndarray
        Input monomial in 2d array format.

    Returns
    -------
    numpy.ndarray
        Reversed monomial.
    """
    return np.flipud(mon)


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def to_name(monomial: np.ndarray,
            names: List[str]) -> str:
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
def nb_all_commuting(mon: np.ndarray,
                     lexorder: np.ndarray,
                     notcomm: np.ndarray) -> bool_:
    """Check if all operators in ``mon`` commute.

    Parameters
    ----------
    mon : numpy.ndarray
        Input monomial in 2d array format.
    lexorder : numpy.ndarray
        A matrix where each row is an operator, and the `i`-th row stores the
        operator with lexicographic rank `i`.
    notcomm : numpy.ndarray
        Matrix of commutation relations. Each operator can be identified by an
        integer `i` which also doubles as its lexicographic rank. Given two
        operators with ranks `i`, `j`, ``notcomm[i, j]`` is 1 if the operators
        do not commute, and 0 if they do.

    Returns
    -------
    bool
        Return `True` if all operators commute, and `False` otherwise.
    """
    if len(mon) <= 1:
        return True
    lexmon = nb_mon_to_lexrepr(mon, lexorder)
    sub_notcomm = notcomm[lexmon, :][:, lexmon]
    return not sub_notcomm.any()


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
        The permutation of the copies of the specified source. The format for
        the permutation here is to use indexing starting at one, so the
        permutation must be padded with a leading zero. The function
        ``causalinflation.quantum.quantum_tools.format_permutations`` converts
        a list of permutations to the necessary format.

    Returns
    -------
    numpy.ndarray
        Input monomial with the specified source swapped.
    """
    new_factors = monomial.copy()
    new_factors[:, 1 + source] = permutation[new_factors[:, 1 + source]]
    return new_factors


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def commutation_matrix(lexorder: np.ndarray,
                       commuting=False) -> np.ndarray:
    """Build a matrix encoding of which operators commute according to the
    function ``nb_operators_commute``. Rows and columns are indexed by
    operators, and each element is a 0 if the operators in the row and in the
    column commute or 1 if they do not.

    Parameters
    ----------
    lexorder : numpy.ndarray
        Matrix with rows as operators where the index of the row gives the
        lexicographic order of the operator.

    commuting : bool
        Whether all the monomials commute. In such a case, the trivial all-zero
        array is returned.

    Returns
    -------
    numpy.ndarray
        Matrix whose entry :math:`(i,j)` has value 1 if the operators with
        lexicographic ordering :math:`i` and :math:`j` do not commute, and
        value 0 if they commute.
    """
    notcomm = np.zeros((lexorder.shape[0], lexorder.shape[0]), dtype=bool_)
    if not commuting:
        for i in range(lexorder.shape[0]):
            for j in range(i + 1, lexorder.shape[0]):
                notcomm[i, j] = not nb_operators_commute(lexorder[i],
                                                         lexorder[j])
        notcomm = notcomm + notcomm.T
    return notcomm


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_inf_indices_refer_common_source(op1_inf_indxs: np.ndarray,
                                       op2_inf_indxs: np.ndarray) -> bool_:
    """Determine if there is any common source referred to by two
     specifications of inflation (a.k.a. 'copy') indices.

    Parameters
    ----------
    op1_inf_indxs : numpy.ndarray
        An array of integers indicating which copy index of each source is
         being referenced.
    op2_inf_indxs : numpy.ndarray
        An array of integers indicating which copy index of each source is
         being referenced.

    Returns
    -------
    bool
        ``True`` if ``op1_inf_indxs`` and ``op2_inf_indxs`` share a source in
        common, ``False`` if they do not.
    """
    common_active_source_types = np.logical_and(op1_inf_indxs, op2_inf_indxs)
    if not np.any(common_active_source_types):
        return False
    return not np.subtract(op1_inf_indxs[common_active_source_types],
                           op2_inf_indxs[common_active_source_types]).all()


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_operators_commute(operator1: np.ndarray,
                         operator2: np.ndarray) -> bool_:
    """Determine if two operators commute. Currently, this only takes into
    account commutation coming from inflation and settings.

    Parameters
    ----------
    operator1 : numpy.ndarray
        Operator as an array of integers.
    operator2 : numpy.ndarray
        Operator as an array of integers.

    Returns
    -------
    bool
        ``True`` if ``operator1`` and ``operator2`` commute, ``False`` if they
        do not.

    Examples
    --------
    A^11_00 commutes with A^22_00
    >>> nb_operators_commute(np.array([1, 1, 1, 0, 0]),
                             np.array([1, 2, 2, 0, 0]))
    True

    A^11_00 does not commute with A^12_00 because they overlap on source 1.
    >>> nb_operators_commute(np.array([1, 1, 1, 0, 0]),
                             np.array([1, 1, 2, 0, 0]))
    False
    """
    if operator1[0] != operator2[0]:  # Different parties
        return True
    if np.array_equal(operator1[1:-1], operator2[1:-1]):  # sources & settings
        return True
    return not nb_inf_indices_refer_common_source(operator1[1:-2],
                                                  operator2[1:-2])


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_lexmon_to_canonical(lexmon: np.ndarray,
                           notcomm: np.ndarray) -> np.ndarray:
    """Brings a monomial, input as the indices of the operators in the
    lexicographic ordering, to canonical form with respect to commutations.

    Parameters
    ----------
    lexmon : numpy.ndarray
        Monomial as an array of integers, where each integer represents
        the lexicographic order of an operator.
    notcomm : numpy.ndarray
        Matrix of commutation relations, given in the format specified by
        `causalinflation.quantum.fast_npa.commutation_matrix`.

    Returns
    -------
    numpy.ndarray
        Monomial in canonical form with respect to commutations.
    """
    if lexmon.shape[0] <= 1:
        return lexmon

    # Take only the rows and columns of notcomm that appear in the monomial,
    # in the correct order.
    sub_notcomm = notcomm[lexmon, :][:, lexmon]
    if not sub_notcomm.any():
        return np.sort(lexmon)
    if sub_notcomm.all():
        return lexmon
    minimum = lexmon[0]
    minimum_idx = 0
    for op in range(1, lexmon.shape[0]):
        # Find the lowest position where we can move op
        nc_ops_position = np.where(sub_notcomm[op, :op] == 1)[0]
        if nc_ops_position.size < 1:
            if lexmon[op] < minimum:
                minimum_idx = op
                minimum     = lexmon[op]
    if minimum <= lexmon[0]:
        m1 = np.array([lexmon[minimum_idx]])
        m2 = np.concatenate((lexmon[:minimum_idx],
                             lexmon[minimum_idx + 1:]))
        return np.concatenate((m1, nb_lexmon_to_canonical(m2, notcomm)))
    else:
        m1 = np.array([lexmon[0]])
        m2 = lexmon[1:]
        return np.concatenate((m1, nb_lexmon_to_canonical(m2, notcomm)))


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def to_canonical(mon: np.ndarray,
                 notcomm: np.ndarray,
                 lexorder: np.ndarray,
                 commuting=False,
                 apply_only_commutations=False) -> np.ndarray:
    """Brings a monomial to canonical form with respect to commutations,
    removing square projectors, and identifying orthogonality.

    Parameters
    ----------
    mon : numpy.ndarray
        Input monomial in 2D array format.
    notcomm : numpy.ndarray
        Matrix of commutation relations. Each operator can be identified by an
        integer `i` which also doubles as its lexicographic rank. Given two
        operators with ranks `i`, `j`, ``notcomm[i, j]`` is 1 if the operators
        do not commute, and 0 if they do.
    lexorder : numpy.ndarray
        A matrix where each row is an operator, and the `i`-th row stores
        the operator with lexicographic rank `i`.
    commuting : bool, optional
        Whether all the variables in the problem commute or not. By default
        ``False``.
    apply_only_commutations : bool, optional
        If ``True``, skip the removal of projector squares and the test to see
        if the monomial is equal to zero. By default ``False``.

    Returns
    -------
    numpy.ndarray
        The monomial in canonical form with respect to some commutation
        relationships.
    """
    mon = np.asarray(mon, dtype=uint8_)
    if mon.shape[0] <= 1:
        return mon
    else:
        mon = order_via_commutation(mon, notcomm, lexorder, commuting)
        if apply_only_commutations:
            return mon
        else:
            mon = remove_projector_squares(mon)
            if mon_is_zero(mon):
                return np.asarray(0*mon[:1], dtype=uint8_)
            else:
                return mon


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def order_via_commutation(mon: np.ndarray,
                          notcomm: np.ndarray,
                          lexorder: np.ndarray,
                          commuting=False) -> np.ndarray:
    """Applies commutations between the operators forming a monomial until
    finding the smallest lexicographic representation.

    Parameters
    ----------
    mon : numpy.ndarray
        Input monomial in 2D array format.
    notcomm : numpy.ndarray
        Matrix of commutation relations. Each operator can be identified by an
        integer `i` which also doubles as its lexicographic rank. Given two
        operators with ranks `i`, `j`, ``notcomm[i, j]`` is 1 if the operators
        do not commute, and 0 if they do.
    lexorder : numpy.ndarray
        A matrix where each row is an operator, and the `i`-th row stores
        the operator with lexicographic rank `i`.
    commuting : bool, optional
        Whether all the variables in the problem commute or not. By default
        ``False``.

    Returns
    -------
    numpy.ndarray
        The monomial in canonical form with respect to some commutation
        relationships.
    """
    mon = np.asarray(mon, dtype=uint8_)
    if len(mon) <= 1:
        return mon
    else:
        if commuting:
            mon = mon_lexsorted(mon, lexorder)
            return mon
        else:
            lexmon = nb_mon_to_lexrepr(mon, lexorder)
            mon = nb_lexmon_to_canonical(lexmon, notcomm)
            mon = lexorder[mon]
            return mon
