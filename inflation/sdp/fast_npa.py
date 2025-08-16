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
    from numba.types import intc as int_
    nopython = True
except ImportError:
    print("Warning: NOT using Numba for JIT compilation.")
    def jit(*args, **kwargs):
        return lambda f: f
    bool_    = bool
    nopython = False
    uint8_   = np.uint8
    int_     = np.intc
    void     = None

cache    = True
if not nopython:
    from scipy.sparse.csgraph import connected_components
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
def nb_classify_disconnected_components(adj_mat: np.ndarray) -> np.ndarray:
    """Given a boolean matrix where each cell indicates whether the supports of
    the operator denoting the row and the operator denoting the column overlap,
    generate a list determining to which disconnected component each operator
    belongs to.

    Parameters
    ----------
    adj_mat : numpy.ndarray
        Boolean 2d array where each cell indicates whether the supports of the
        operator denoting the row and the operator denoting the column overlap.

    Returns
    -------
    numpy.ndarray
        A list of integers of size the number of operators used for creating
        adj_mat, where each integer indexes the disconnected component the
        corresponding operator belongs to.
    """
    if not nopython:
        return connected_components(adj_mat,
                                    directed=False,
                                    return_labels=True)[-1]
    # See https://stackoverflow.com/a/9112588 for inspiration of the method
    n = len(adj_mat)
    if n <= 1 or adj_mat.all():
        return np.zeros((n,), dtype=uint8_)
    component_labels = np.zeros((n,), dtype=uint8_)
    component_counter = 1
    for i in range(n):
        if not component_labels[i]:
            old_component = np.logical_not(np.ones((n,), dtype=uint8_))
            new_component = old_component.copy()
            new_component[i] = True
            search_next = np.logical_xor(new_component, old_component)
            while search_next.any():
                old_component = new_component.copy()
                new_component = np.logical_or(
                    new_component,
                    adj_mat[search_next].sum(axis=0).astype(bool_))
                search_next = np.logical_xor(new_component, old_component)
            component_labels[new_component] = component_counter
            component_counter += 1
    return (component_labels-1).astype(np.uint8)


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_is_knowable(monomial: np.ndarray) -> bool_:
    """Determine whether a given atomic monomial admits an identification with
    a probability of the original scenario.

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
    # Knowable monomials have at most one operator per party and one copy of
    # each source in the DAG
    parties = monomial[:, 0]
    if len(np.unique(parties)) != len(monomial):
        return False
    for source in monomial.T[1:-2]:
        if len(np.unique(source[np.flatnonzero(source)])) > 1:
            return False
    return True


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
    return -1

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
    lex = np.empty(mon.shape[0], dtype=int_)
    for i in range(mon.shape[0]):
        lex[i] = nb_lexorder_idx(mon[i], lexorder)
    return lex


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_monomial_to_components(monomial: np.ndarray) -> np.ndarray:
    """Wrapper for obtaining the list of disconnected components of a monomial.

    Parameters
    ----------
    monomial : numpy.ndarray
        Monomial in 2d array form.

    Returns
    -------
    numpy.ndarray
        A vector where each integer gives the component associated with the
        operator of that index.

    Examples
    --------
    >>> monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [2, 1, 0, 2, 0, 0],
                             [1, 0, 3, 3, 0, 0],
                             [3, 3, 5, 0, 0, 0],
                             [3, 1, 4, 0, 0, 0],
                             [3, 6, 6, 0, 0, 0],
                             [3, 4, 5, 0, 0, 0]])
    >>> factorised = nb_monomial_to_components(monomial)
    [0, 1, 2, 3, 1, 4, 3]
    """
    n = len(monomial)
    if n <= 1:
        return np.zeros(n, dtype=uint8_)
    return nb_classify_disconnected_components(nb_overlap_matrix(
        monomial[:, 1:-2]))


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_overlap_matrix(inflation_indxs: np.ndarray) -> np.ndarray:
    """Given a list of inflation indices for a number of operators, generate
    a boolean matrix whose entries denote whether the supports of the operator
    indexed by the row and of the operator indexed by the column overlap.

    Parameters
    ----------
    inflation_indxs : numpy.ndarray
        A list of inflation indices for a collection of operators.

    Returns
    -------
    numpy.ndarray
        The "adjacency matrix" whose entries denote whether the supports of the
        operator indexed by the row and of the operator indexed by the column
        overlap.
    """
    n = len(inflation_indxs)
    adj_mat = np.eye(n, dtype=bool_)
    for i in range(1, n):
        inf_indices_i = inflation_indxs[i]
        for j in range(i):
            inf_indices_j = inflation_indxs[j]
            if nb_exists_shared_source(inf_indices_i, inf_indices_j):
                adj_mat[i, j] = True
    adj_mat = np.logical_or(adj_mat, adj_mat.T)
    return adj_mat


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_indices_of_more_than_one_op_per_party_per_factor(monomial: np.ndarray,
                                                        sandwich_positivity=True) -> np.ndarray:
    r"""If sandwich_positivity=True, this removes sandwiching/pinching from the
    monomial. This is, it converts the monomial represented by
    :math:`U A U^\dagger` into :math:`A`.
    Regardless, it then factorises the monomial into components, yielding
    a boolean vector where True indicates an operator which appears in the same
    factor as another of the same party.

    Parameters
    ----------
    monomial : numpy.ndarray
        Input monomial.
    sandwich_positivity : bool, optional
        Whether to consider sandwiching. By default ``True``.

    Returns
    -------
    numpy.ndarray
        The monomial without sandwiches.
    """
    picklist = np.logical_not(np.ones(len(monomial)))
    parties = np.unique(monomial[:, 0])
    for party in parties:
        indices_for_this_party = np.flatnonzero(monomial[:, 0] == party)
        party_monomial = monomial[indices_for_this_party]
        party_monomial_comp_labels = nb_monomial_to_components(party_monomial)
        for i in range(party_monomial_comp_labels.max() + 1):
            indices_for_this_factor = np.flatnonzero(
                party_monomial_comp_labels == i)
            factor = party_monomial[indices_for_this_factor]
            # We now only return factors with more than one operator for a single party!
            if sandwich_positivity:
                while len(factor) and np.array_equal(factor[0], factor[-1]):
                    indices_for_this_factor = indices_for_this_factor[1:-1]
                    factor = factor[1:-1]
            if len(factor) > 1:
                picklist[indices_for_this_party[indices_for_this_factor]] = True
    return picklist

@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_is_physical(monomial_in: np.ndarray, sandwich_positivity=True) -> bool_:
    r"""Determines whether a monomial is physical, this is, if it always has a
    non-negative expectation value.

    This code also supports the detection of "sandwiches", i.e., monomials
    of the form :math:`\langle \psi | A_1 A_2 A_1 | \psi \rangle` where
    :math:`A_1` and :math:`A_2` do not commute. In principle we do not know the
    value of this term. However, note that :math:`A_1` can be absorbed into
    :math:`| \psi \rangle` forming an unnormalised quantum state
    :math:`| \psi' \rangle`, thus :math:`\langle\psi'|A_2|\psi'\rangle`.
    Note that while we know the value :math:`\langle\psi |A_2| \psi\rangle`,
    we do not know :math:`\langle \psi' | A_2 | \psi' \rangle` because of
    the unknown normalisation, however we know it must be non-negative,
    :math:`\langle \psi | A_1 A_2 A_1 | \psi \rangle \geq 0`.
    This simple example can be extended to various layers of sandwiching.

    Parameters
    ----------
    monomial_in : numpy.ndarray
        Input monomial in 2d array format.
    sandwich_positivity : bool, optional
        Whether to consider sandwiching. By default ``True``.

    Returns
    -------
    bool
        Whether the monomial has always non-negative expectation or not.
    """
    return not nb_indices_of_more_than_one_op_per_party_per_factor(
        monomial_in, sandwich_positivity=sandwich_positivity).any()

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
# @jit(nopython=nopython, cache=cache, forceobj=not nopython)
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
        ``inflation.sdp.quantum_tools.format_permutations`` converts
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
                       sources_to_check_for_pairwise: np.ndarray,
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
    quantum_sources: numpy.ndarray
        List of integers denoting the columns that in the 2D array encoding
        corresponding to inflation indices of quantum sources (as opposed to
        classical sources). Example: np.array([1, 3]) implies that the 1st
        and 3rd columns of an operator in 2D array form correspond to inflation
        indices that act on quantum sources.
    sources_to_check_for_pairwise: numpy.ndarray
        A 3-index tensor boolean array which keep track of which pairs of parties
        have which quantum sources in common via intermediate quantum latents.
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
                                                         lexorder[j],
                                                         sources_to_check_for_pairwise)
        notcomm = notcomm + notcomm.T
    return notcomm


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_all_commuting_q(mon: np.ndarray,
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
        Return ``True`` if all operators commute, and ``False`` otherwise.
    """
    if len(mon) <= 1:
        return True
    lexmon = nb_mon_to_lexrepr(mon, lexorder)
    sub_notcomm = notcomm[lexmon, :][:, lexmon]
    return not sub_notcomm.any()


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_exists_shared_source(inf_indices1: np.ndarray,
                            inf_indices2: np.ndarray) -> bool_:
    """Determine if there is any common source referred to by two
     specifications of inflation (a.k.a. 'copy') indices.

    Parameters
    ----------
    inf_indices1 : numpy.ndarray
        An array of integers indicating which copy index of each source is
        being referenced.
    inf_indices2 : numpy.ndarray
        An array of integers indicating which copy index of each source is
        being referenced.

    Returns
    -------
    bool
        ``True`` if ``inf_indices1`` and ``inf_indices2`` share a source in
        common, ``False`` if they do not.
    """
    common_sources = np.logical_and(inf_indices1, inf_indices2)
    if not np.any(common_sources):
        return False
    return not np.subtract(inf_indices1[common_sources],
                           inf_indices2[common_sources]).all()
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
        `inflation.sdp.fast_npa.commutation_matrix`.

    Returns
    -------
    numpy.ndarray
        Monomial in canonical form with respect to commutations.
    """
    result = np.empty((0,), dtype=int_)
    leftover = lexmon
    while leftover.shape[0] > 1:
        lexmon = leftover
        ########################################################################
        # A subroutine that splits lexmon into two monomials, m1 and m2,
        # such that m1 is appended to result, and m2 is the new lexmon for a
        # new iteration of the while loop.
        # This subroutine should be defined in a function but numba
        # gives problems with returning tuples (m1,m2) from functions.
        if lexmon.shape[0] <= 1:
            m1, m2 = lexmon, np.empty((0,), dtype=int_)
        else:
            # Take only the rows and columns of notcomm that appear in the
            # monomial, in the correct order.
            sub_notcomm = notcomm[lexmon, :][:, lexmon]
            if sub_notcomm.all():
                m1, m2 = lexmon, np.empty((0,), dtype=int_)
            elif not sub_notcomm.any():
                m1, m2 = np.sort(lexmon), np.empty((0,), dtype=int_)
            else:
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
                else:
                    m1 = np.array([lexmon[0]])
                    m2 = lexmon[1:]
        ########################################################################
        result = np.concatenate((result, m1))
        leftover = m2
    # Depending on how the while loop ends, we may have a leftover monomial
    if leftover.shape[0] != 0:
        result = np.concatenate((result, leftover))
    return result

@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def nb_operators_commute(operator1: np.ndarray,
                         operator2: np.ndarray,
                         sources_to_check_for_pairwise: np.ndarray
                         ) -> bool_:
    """Determine if two operators commute. Currently, this only takes into
    account commutation coming from inflation and settings.

    Parameters
    ----------
    operator1 : numpy.ndarray
        Operator as an array of integers.
    operator2 : numpy.ndarray
        Operator as an array of integers.
    sources_to_check_for_pairwise: numpy.ndarray
        A 3-index tensor boolean array which keep track of which pairs of parties
        have which quantum or classical sources in common via intermediate latents.

    Returns
    -------
    bool
        ``True`` if ``operator1`` and ``operator2`` commute, ``False`` if they
        do not.

    Examples
    --------
    A^11_00 does not commute with A^12_00 because they overlap on source 1.
    >>> nb_operators_commute(np.array([1, 1, 1, 0, 0]),
                             np.array([1, 1, 2, 0, 0]),  np.array([[[2,2],[0,0]],[[0,0],[2,2]]))
    False
    
    A^11_00 commutes with A^12_00 because source 1 is classical.
    >>> nb_operators_commute(np.array([1, 1, 1, 0, 0]),
                             np.array([1, 1, 2, 0, 0]),  np.array([[[1,2],[0,0]],[[0,0],[1,2]]))
    True
    """
    # Case 0: Different parties commute
    party1 = operator1[0] - 1
    party2 = operator2[0] - 1
    relevant_sources = sources_to_check_for_pairwise[party1, party2]
    common_source_positions_all = np.greater_equal(relevant_sources, 1)
    common_source_positions_quantum = np.greater_equal(relevant_sources, 2)

    # Case 1: no common quantum source TYPE
    if not common_source_positions_quantum.any():
        return True
    # Case 2: yes common quantum source TYPE, but different INDICES across all quantum source
    sources1 = operator1[1:-2]
    sources2 = operator2[1:-2]
    if np.subtract(sources1[common_source_positions_quantum],
                   sources2[common_source_positions_quantum]).all():
        return True
    # Case 3: overlapping quantum source type and index, but not all sources match indices
    if not np.array_equal(sources1[common_source_positions_all],
                          sources2[common_source_positions_all]):
        return False
    # Case 4: All sources of type common to both operators match indices
    # # Case 4a: different parties
    if party1 != party2:
        return True
    # Case 4b: Same party (check the settings)
    return operator1[-2] == operator2[-2]

# Node: cache is set to False because of a bug in numba, which affects
# nb_lexmon_to_canonical which this function uses
@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def to_canonical_1d_internal(lexmon: np.ndarray,
                             notcomm: np.ndarray,
                             orthomat: np.ndarray,
                             commuting=False,
                             apply_only_commutations=False) -> np.ndarray:
    """Brings a monomial to canonical form with respect to commutations,
    removing square projectors, and identifying orthogonality.

    Parameters
    ----------
    lexmon : numpy.ndarray
        Input monomial in 1D array format.
    notcomm : numpy.ndarray
        Matrix of commutation relations. Each operator can be identified by an
        integer `i` which also doubles as its lexicographic rank. Given two
        operators with ranks `i`, `j`, ``notcomm[i, j]`` is 1 if the operators
        do not commute, and 0 if they do.
    orthomat : numpy.ndarray
        Matrix of orthogonality relations. Each operator can be identified by an
        integer `i` which also doubles as its lexicographic rank. Given two
        operators with ranks `i`, `j`, ``notcomm[i, j]`` is 1 if the operators
        are orthogonal, and 0 if they do.
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
    if len(lexmon) <= 1:
        return lexmon
    else:
        if commuting:
            newlexmon = np.sort(lexmon)
        else:
            newlexmon = nb_lexmon_to_canonical(lexmon, notcomm)
    if apply_only_commutations:
        return newlexmon
    else:
        if lexmon_is_zero(newlexmon, orthomat):
            newlexmon = np.zeros((1,), dtype=int_)
        return remove_projector_squares(newlexmon)


@jit(nopython=nopython, cache=cache, forceobj=not nopython)
def lexmon_is_zero(lexmon: np.ndarray, orthomat: np.ndarray) -> bool_:
    if np.array_equal(lexmon, [0]):
        return True
    for i in range(1, lexmon.shape[0]):
        if orthomat[lexmon[i], lexmon[i-1]]:
            return True
    return False
