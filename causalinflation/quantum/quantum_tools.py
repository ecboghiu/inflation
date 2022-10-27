"""
This file contains helper functions to manipulate monomials and generate moment
matrices.
@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""
import numpy as np
import sympy

from copy import deepcopy
from itertools import permutations, product
from typing import Dict, Iterable, List, Tuple

from .fast_npa import (apply_source_perm,
                       dot_mon,
                       mon_is_zero,
                       mon_lexsorted,
                       to_canonical,
                       to_name)

try:
    from tqdm import tqdm
except ImportError:
    from ..utils import blank_tqdm as tqdm


###############################################################################
# FUNCTIONS FOR MONOMIALS                                                     #
###############################################################################
def factorize_monomial(raw_monomial: np.ndarray,
                       canonical_order=False) -> Tuple[np.ndarray]:
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
    raw_monomial : np.ndarray
        Monomial in 2d array form.
    canonical_order: bool, optional
        Whether to return the different factors in a canonical order.

    Returns
    -------
    Tuple[np.ndarray]
        A tuple of ndarrays, where each array represents an atomic monomial
        factor.

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
        return (raw_monomial,)

    monomial = np.asarray(raw_monomial, dtype=np.uint16)
    components_indices = np.zeros((len(monomial), 2), dtype=np.uint16)
    # Labels to see if the components have been used
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
                    overlapping = (inflation_indices[:, source]
                                   == inflation_indices[component[jdx], source]
                                   )
                    # Add the components that overlap to the lookup list
                    component += components_indices[
                                  overlapping & (components_indices[:, 1] == 0)
                                                    ][:, 0].tolist()
                    # Specify that the components that overlap have been used
                    components_indices[overlapping, 1] = 1
                jdx += 1
        if len(component) > 0:
            disconnected_components.append(component)
        idx += 1

    disconnected_components = tuple(
        monomial[sorted(component)] for component in disconnected_components)

    if canonical_order:
        disconnected_components = tuple(sorted(disconnected_components,
                                               key=lambda x: x.tobytes()))
    return disconnected_components


def flatten_symbolic_powers(monomial: sympy.core.symbol.Symbol
                            ) -> List[sympy.core.symbol.Symbol]:
    """If we have powers of a monomial, such as A**3, return a list with
    the factors, [A, A, A].

    Parameters
    ----------
    monomial : sympy.core.symbol.Symbol
        Symbolic monomial, possible with powers.

    Returns
    -------
    List[sympy.core.symbol.Symbol]
        List of all the symbolic factors, with the powers expanded.
    """
    factors          = monomial.as_ordered_factors()
    factors_expanded = []
    for factor in factors:
        base, exp = factor.as_base_exp()
        if exp == 1:
            factors_expanded.append(base)
        elif exp > 1:
            for _ in range(exp):
                factors_expanded.append(base)
    factors = factors_expanded
    return factors


def is_knowable(monomial: np.ndarray) -> bool:
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
    assert monomial.ndim == 2, \
        ("You must enter a list of operators. Hence, the number of dimensions "
         + "of the monomial must be 2")
    parties = monomial[:, 0].astype(int)
    # Knowable monomials have at most one operator per party and one copy of
    # each source in the DAG
    if len(set(parties)) != len(parties):
        return False
    else:
        return all([len(set(source[np.nonzero(source)])) <= 1
                    for source in monomial[:, 1:-2].T])


def is_physical(monomial_in: np.ndarray,
                sandwich_positivity=True
                ) -> bool:
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
    if len(monomial_in) <= 1:
        return True
    if sandwich_positivity:
        monomial = remove_sandwich(monomial_in)
        if len(monomial) <= 1:
            return True
    else:
        monomial = monomial_in
    parties = np.unique(monomial[:, 0])
    for party in parties:
        party_monomial = monomial[monomial[:, 0] == party]
        n = len(party_monomial)
        if not n == 1:
            factors = factorize_monomial(party_monomial)
            if len(factors) != n:
                return False
    return True


def reduce_inflation_indices(monomial: np.ndarray) -> np.ndarray:
    """Reduce the inflation indices of a monomial as much as possible. This
    procedure might not give the canonical form directly due to commutations.

    Parameters
    ----------
    monomial : numpy.ndarray
        Input monomial.

    Returns
    -------
    numpy.ndarray
        An equivalent monomial closer to its representative form.
    """
    new_mon    = monomial.copy()
    nr_sources = monomial.shape[1] - 3
    # Pad the monomial with a row a zeros at the front so as to have the
    # relevant inflation indices to begin at 1
    new_mon_padded_transposed = np.concatenate((np.zeros_like(new_mon[:1]),
                                                new_mon)).T
    for source in range(nr_sources):
        copies_used = new_mon_padded_transposed[1 + source]
        _, unique_positions = np.unique(copies_used, return_inverse=True)
        new_mon_padded_transposed[1 + source] = unique_positions
    return new_mon_padded_transposed.T[1:]


def remove_sandwich(monomial: np.ndarray) -> np.ndarray:
    r"""Removes sandwiching/pinching from a monomial. This is, it converts the
    monomial represented by :math:`U A U^\dagger` into :math:`A`.

    Parameters
    ----------
    monomial : numpy.ndarray
        Input monomial.

    Returns
    -------
    numpy.ndarray
        The monomial without sandwiches.
    """
    new_monomial = np.empty((0, monomial[0, :].shape[0]), dtype=int)
    parties = np.unique(monomial[:, 0])
    for party in parties:
        party_monomial = monomial[monomial[:, 0] == party]
        party_monomial_factorized = factorize_monomial(party_monomial)
        for factor in party_monomial_factorized:
            while (len(factor) > 1) and np.array_equal(factor[0], factor[-1]):
                factor = factor[1:-1]
            new_monomial = np.append(new_monomial, factor, axis=0)
    return new_monomial


###############################################################################
# FUNCTIONS FOR MOMENT MATRICES                                               #
###############################################################################
def calculate_momentmatrix(cols: List,
                           notcomm: np.ndarray,
                           lexorder: np.ndarray,
                           commuting: bool = False,
                           verbose: int = 0) -> Tuple[np.ndarray, Dict]:
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

    Returns
    -------
    Tuple[numpy.ndarray, Dict]
        The moment matrix :math:`\Gamma`, where each entry :math:`(i,j)` stores
        the integer representation of a monomial. The Dict is a mapping from
        string representation to integer representation.
    """
    nrcols = len(cols)
    canonical_mon_to_idx = dict()
    momentmatrix = np.zeros((nrcols, nrcols), dtype=np.uint32)
    varidx = 1  # We start from 1 because 0 is reserved for 0
    for i, mon1 in tqdm(enumerate(cols),
                  disable=not verbose,
                  desc="Calculating moment matrix",
                  total=nrcols):
        for j in range(i, nrcols):
            mon2 = cols[j]
            mon_v1 = to_canonical(dot_mon(mon1, mon2),
                                  notcomm,
                                  lexorder,
                                  commuting=commuting)
            if not mon_is_zero(mon_v1):
                if not commuting:
                    mon_v2 = to_canonical(dot_mon(mon2, mon1),
                                          notcomm,
                                          lexorder,
                                          commuting=commuting)
                    mon_hash = min(mon_v1.tobytes(), mon_v2.tobytes())
                else:
                    mon_hash = mon_v1.tobytes()
                try:
                    known_varidx = canonical_mon_to_idx[mon_hash]
                    momentmatrix[i, j] = known_varidx
                    momentmatrix[j, i] = known_varidx
                except KeyError:
                    canonical_mon_to_idx[mon_hash] = varidx
                    momentmatrix[i, j] = varidx
                    momentmatrix[j, i] = varidx
                    varidx += 1
    return momentmatrix, canonical_mon_to_idx


###############################################################################
# FUNCTIONS FOR INFLATIONS                                                    #
###############################################################################
def apply_inflation_symmetries(momentmatrix: np.ndarray,
                               inflation_symmetries: np.ndarray,
                               verbose: bool = False
                               ) -> Tuple[np.ndarray,
                                          Dict[int, int],
                                          np.ndarray]:
    """Applies the inflation symmetries, in the form of permutations of the
    rows and colums of a moment matrix, to the moment matrix.

    Parameters
    ----------
    momentmatrix : numpy.ndarray
        The moment matrix.
    inflation_symmetries : numpy.ndarray
        Two-dimensional array where each row represents a permutation of
        the rows and columns of the moment matrix.
    verbose : bool
        Whether information about progress is printed out.

    Returns
    -------
    sym_mm : numpy.ndarray
        The symmetrized version of the moment matrix, where each cell is
        the lowest index of all the Monomials that are equivalent to that
        in the corresponding cell in momentmatrix.
    orbits : Dict[int, int]
        The map from unsymmetrized indices in momentmatrix to their
        symmetrized counterparts in sym_mm.
    repr_values: numpy.ndarray
        An array of unique representative former (unsymmetrized) indices.
        This is later used for hashing indices and making sanitization much
        faster.
    """
    max_value = momentmatrix.max(initial=0)
    if not len(inflation_symmetries):
        repr_values = np.arange(max_value + 1)
        orbits = dict(zip(repr_values, repr_values))
        return momentmatrix, orbits, repr_values
    else:
        old_indices, flat_pos, inverse = np.unique(momentmatrix.ravel(),
                                                   return_index=True,
                                                   return_inverse=True)
        inverse           = inverse.reshape(momentmatrix.shape)
        prev_unique_count = np.inf
        new_unique_count  = old_indices.shape[0]
        new_indices       = np.arange(new_unique_count)
        inversion_tracker = new_indices.copy()
        repr_values = old_indices.copy()
        # We minimize under every element of the inflation symmetry group.
        for permutation in tqdm(inflation_symmetries,
                                disable=not verbose,
                                desc="Applying symmetries      "):
            if prev_unique_count > new_unique_count:
                rows, cols = np.unravel_index(flat_pos, momentmatrix.shape)
            prev_unique_count = new_unique_count
            assert np.array_equal(new_indices, inverse[(rows, cols)]), \
                ("The representatives of the symmetrized indices are " +
                 "not minimal.")
            np.minimum(new_indices,
                       inverse[(permutation[rows], permutation[cols])],
                       out=new_indices)
            unique_values, unique_values_pos, unique_values_inv = \
                np.unique(new_indices,
                          return_index=True,
                          return_inverse=True)
            new_unique_count = unique_values.shape[0]
            if prev_unique_count > new_unique_count:
                inverse           = unique_values_inv[inverse]
                flat_pos          = flat_pos[unique_values_pos]
                repr_values       = repr_values[unique_values_pos]
                inversion_tracker = unique_values_inv[inversion_tracker]
                del unique_values_pos, unique_values_inv
                new_indices = np.arange(new_unique_count)
        prior_min_value = old_indices.min()
        if old_indices.min() != 0:
            new_indices += prior_min_value
        orbits = dict(zip(old_indices, new_indices[inversion_tracker]))
        sym_mm = new_indices[inverse]
        return sym_mm, orbits, repr_values


def commutation_relations(infSDP):
    """Return a user-friendly representation of the commutation relations.

    Parameters
    ----------
    infSDP : causalinflation.InflationSDP
        The SDP object for which the commutation relations are to be extracted.

    Returns
    -------
    Tuple[sympy.Expr]
        The list of commutators (given as sympy Expressions) that are nonzero.
    """
    from collections import namedtuple
    nonzero = namedtuple("NonZeroExpressions", "exprs")
    data = []
    for i in range(infSDP._lexorder.shape[0]):
        for j in range(i, infSDP._lexorder.shape[0]):
            # Most operators commute as they belong to different parties,
            if infSDP._notcomm[i, j] != 0:
                op1 = sympy.Symbol(to_name([infSDP._lexorder[i]],
                                           infSDP.names),
                                   commutative=False)
                op2 = sympy.Symbol(to_name([infSDP._lexorder[i]],
                                           infSDP.names),
                                   commutative=False)
                if infSDP.verbose > 0:
                    print(f"{str(op1 * op2 - op2 * op1)} ≠ 0.")
                data.append(op1 * op2 - op2 * op1)
    return nonzero(data)


def construct_normalization_eqs(column_equalities: List[Tuple[int, List[int]]],
                                momentmatrix: np.ndarray,
                                verbose=0,
                                ) -> List[Tuple[int, List[int]]]:
    """Given a list of column level normalization equalities and the moment
    matrix, this function computes the implicit normalization equalities
    between matrix elements. Column-level and monomial-level equalities share
    nearly the same format, they differ merely in whether integers pertain to
    column indices or the indices that represent the unique moment matrix
    elements.
    """
    equalities = []
    seen_already = set()
    nof_seen_already = len(seen_already)
    for equality in tqdm(column_equalities,
                         disable=not verbose,
                         desc="Imposing normalization   "):
        for i, row in enumerate(iter(momentmatrix)):
            (normalization_col, summation_cols) = equality
            norm_idx       = row[normalization_col]
            summation_idxs = row[summation_cols]
            if summation_idxs.all():
                summation_idxs.sort()
                seen_already.add(tuple(summation_idxs.flat))
                if len(seen_already) > nof_seen_already:
                    equalities.append((norm_idx, summation_idxs.tolist()))
                    nof_seen_already += 1
    del seen_already
    return equalities


def format_permutations(array: np.ndarray) -> np.ndarray:
    """Permutations of inflation indices must leave the integers 0,
    corresponding to sources not being measured by the operator, invariant.
    In order to achieve this, this function shifts a permutation of sources
    by 1 and prepends it with the integer 0.

    Parameters
    ----------
    array : numpy.ndarray
        2-d array where each row is a permutations.

    Returns
    -------
    numpy.ndarray
        The processed list of permutations.
    """
    source_permutation = np.asarray(array) + 1
    padding = np.zeros((len(source_permutation), 1), dtype=int)
    return np.hstack((padding, source_permutation))


def generate_operators(outs_per_input: List[int],
                       name: str
                       ) -> List[List[List[sympy.core.symbol.Symbol]]]:
    """Generates the list of ``sympy.core.symbol.Symbol`` variables
    representing the measurements for a given party. The variables are treated
    as non-commuting. This code is adapted from `ncpol2sdpa
    <https://github.com/peterwittek/ncpol2sdpa/>`_.

    Parameters
    ----------
    outs_per_input : List[int]
        The number of outcomes of each measurement for a given party
    name : str
        The name to be associated to the party

    Returns
    -------
    list
        The list of Sympy operators
    """
    ops_per_input = []
    for x, outs in enumerate(outs_per_input):
        ops_per_output_per_input = []
        for o in range(outs):
            ops_per_output_per_input.append(
                sympy.Symbol(name + "_" + str(x) + "_" + str(o),
                             commutative=False)
            )
        ops_per_input.append(ops_per_output_per_input)
    return ops_per_input


def lexicographic_order(infSDP) -> Dict[str, int]:
    """Return a user-friendly representation of the lexicographic order.

    Parameters
    ----------
    infSDP : causalinflation.InflationSDP
        The SDP object for which the commutation relations are to be extracted.

    Returns
    -------
    Dict[str, int]
        The lexicographic order as a dictionary where keys are the monomials in
        the problem and the values are their positions in the lexicographic
        ordering.
    """
    lexorder = {}
    for i, op in enumerate(infSDP._lexorder):
        lexorder[sympy.Symbol(to_name([op], infSDP.names),
                              commutative=False)] = i
    return lexorder


def party_physical_monomials(hypergraph: np.ndarray,
                             inflevels: np.ndarray,
                             party: int,
                             max_monomial_length: int,
                             settings_per_party: Tuple[int],
                             outputs_per_party: Tuple[int],
                             lexorder: np.ndarray
                             ) -> List[np.ndarray]:
    """Generate all possible non-negative monomials for a given party composed
    of at most ``max_monomial_length`` operators.

    Parameters
    ----------
    hypergraph : numpy.ndarray
         Hypergraph of the scenario.
    inflevels : np.ndarray
        The number of copies of each source in the inflated scenario.
    party : int
        Party index. NOTE: starting from 0
    max_monomial_length : int
        The maximum number of operators in the monomial.
    settings_per_party : List[int]
        List containing the cardinality of the input/measurement setting
        of each party.
    outputs_per_party : List[int]
        List containing the cardinality of the output/measurement outcome
        of each party.
    lexorder : numpy.ndarray
        A matrix storing the lexicographic order of operators. If an operator
        has lexicographic rank `i`, then it is placed at the ``i``-th row of
        lexorder.
        
    Returns
    -------
    List[numpy.ndarray]
        An array containing all possible positive monomials of the given
        length.
    """

    hypergraph = np.asarray(hypergraph)
    nr_sources = hypergraph.shape[0]

    assert max_monomial_length <= min(inflevels), \
        ("You cannot have a longer list of commuting operators" +
         " than the inflation level.")

    # The strategy is building an initial non-negative monomial and apply all
    # inflation symmetries
    initial_monomial = np.zeros(
        (max_monomial_length, 1 + nr_sources + 2), dtype=np.uint16)
    for mon_idx in range(max_monomial_length):
        initial_monomial[mon_idx, 0]    = 1 + party
        initial_monomial[mon_idx, 1:-2] = hypergraph[:, party] * (1 + mon_idx)

    inflation_equivalents = {initial_monomial.tobytes(): initial_monomial}

    all_permutations_per_source = [
        format_permutations(list(permutations(range(inflevel))))
        for inflevel in inflevels.flat]
    for permutation in product(*all_permutations_per_source):
        permuted = initial_monomial.copy()
        for source in range(nr_sources):
            permuted = mon_lexsorted(apply_source_perm(permuted,
                                                       source,
                                                       permutation[source]),
                                     lexorder)
        inflation_equivalents[permuted.tobytes()] = permuted
    inflation_equivalents = list(inflation_equivalents.values())

    new_monomials = []
    # Insert all combinations of inputs and outputs
    for input_slice in product(*[range(settings_per_party[party])
                                 for _ in range(max_monomial_length)]):
        for output_slice in product(*[range(outputs_per_party[party] - 1)
                                      for _ in range(max_monomial_length)]):
            for new_mon_idx in range(len(inflation_equivalents)):
                new_monomial = deepcopy(inflation_equivalents[new_mon_idx])
                for mon_idx in range(max_monomial_length):
                    new_monomial[mon_idx, -2] = input_slice[mon_idx]
                    new_monomial[mon_idx, -1] = output_slice[mon_idx]
                new_monomials.append(new_monomial)
    return new_monomials


###############################################################################
# OTHER FUNCTIONS                                                             #
###############################################################################
def clean_coefficients(cert: Dict[str, float],
                       chop_tol: float = 1e-10,
                       round_decimals: int = 3) -> Dict:
    """Clean the list of coefficients in a certificate.

    Parameters
    ----------
    cert : Dict[str, float]
      A dictionary containing as keys the monomials associated to the elements
      of the certificate and as values the corresponding coefficients.
    chop_tol : float, optional
      Coefficients in the dual certificate smaller in absolute value are
      set to zero. Defaults to ``1e-10``.
    round_decimals : int, optional
      Coefficients that are not set to zero are rounded to the number
      of decimals specified. Defaults to ``3``.

    Returns
    -------
    np.ndarray
      The cleaned-up coefficients.
    """
    processed_cert = deepcopy(cert)
    vars = processed_cert.keys()
    coeffs = np.asarray(list(processed_cert.values()))
    # Take the biggest one and make it 1
    normalising_factor = np.max(np.abs(coeffs[np.abs(coeffs) > chop_tol]))
    coeffs /= normalising_factor
    # Set to zero very small coefficients
    coeffs[np.abs(coeffs) <= chop_tol] = 0
    # Round
    coeffs = np.round(coeffs, decimals=round_decimals)
    return dict(zip(vars, coeffs.flat))
