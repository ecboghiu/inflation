"""
This file contains helper functions to manipulate monomials and generate moment
matrices.
@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""
import numpy as np
import sympy

from copy import deepcopy
from itertools import permutations, product, combinations_with_replacement
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

from .fast_npa import (apply_source_perm,
                       dot_mon,
                       mon_is_zero,
                       mon_lexsorted,
                       to_canonical,
                       to_name)


###############################################################################
# FUNCTIONS FOR MONOMIALS                                                     #
###############################################################################
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
        `inflation.sdp.fast_npa.commutation_matrix`.
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
    for (i, mon1), (j, mon2) in tqdm(
                            combinations_with_replacement(enumerate(cols), 2),
                            disable=not verbose,
                            desc="Calculating moment matrix",
                            total=int(nrcols*(nrcols+1)/2),
                                    ):
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


def to_symbol(mon: np.ndarray,
              names: np.ndarray,
              commutative=False) -> sympy.core.symbol.Symbol:
    """Convert a monomial to a SymPy expression.

    Parameters
    ----------
    mon : numpy.ndarray
        Monomial written as a 2D array.
    names : numpy.ndarray
        The names of the parties in the monomial.
    commutative : bool, optional
        If the operators in the monomial are commutative. By default
        ``False``.

    Returns
    -------
    sp.core.symbol.Symbol
        The monomial as a SymPy expression.

    Examples
    --------
    >>> to_symbol(np.array([[1, 1, 0, 1], [2, 1, 1, 2]]), ["A", "B"])
    A_1_0_1*B_1_1_2
    """
    if isinstance(mon, np.ndarray):
        if mon.shape[0] == 0:
            return sympy.S.One
        res = sympy.S.One
        for mon in mon:
            name = '_'.join([names[mon[0]-1]] +
                            [str(i) for i in mon.tolist()][1:])
            res *= sympy.Symbol(name, commutative=commutative)
        return res


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
    infSDP : inflation.InflationSDP
        The SDP object for which the commutation relations are to be extracted.

    Returns
    -------
    Tuple[sympy.core.expr.Expr]
        The list of commutators (given as sympy Expressions) that are nonzero.
    """
    from collections import namedtuple
    nonzero = namedtuple("NonZeroExpressions", "exprs")
    data = []
    lexorder = infSDP._lexorder
    lexorder_len = lexorder.shape[0]
    for i in range(lexorder_len):
        for j in range(i, lexorder_len):
            # Most operators commute as they belong to different parties,
            if infSDP._notcomm[i, j] != 0:
                (op1_name, op2_name) = to_name(lexorder[[i, j]],
                                               infSDP.names).split('*')
                op1 = sympy.Symbol(op1_name, commutative=False)
                op2 = sympy.Symbol(op2_name, commutative=False)
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

    Parameters
    ----------
    column_equalities : List[Tuple[int, List[int]]]
        The list of equalities between columns in the moment matrix, in the
        form of tuples whose first element is the index of one of the columns,
        and the second element is the list of indices of the columns whose
        corresponding operators sum up to the operator corresponding to the
        first element.
    momentmatrix : numpy.ndarray
        The moment matrix of which the identification between variables shall
        be computed.
    verbose : int, optional
        Verbosity level. By default 0.

    Returns
    -------
    List[Tuple[int, List[int]]]
        The equalities between variables. For each tuple, the first element is
        the index of one of the variables in ``momentmatrix``, and the second
        is the list of variables whose sum corresponds to the first.
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


def expand_moment_normalisation(moment: np.ndarray,
                                outcome_cardinalities: List[int],
                                skip_party: List[bool]):
    """Helper function that identifies operators within the monomial that
    correspond to the last outcome, and uses normalisation to produce
    an equality constraint with other monomials. The constraint is expressed as
    `(i, (i1, i2, i3, ...))`, where the moment corresponding to index `i` is
    equal to the sum of moments corresponding to indices
    `(i1, i2, i3, ...)`.

    Parameters
    ----------
    moment : numpy.ndarray
        Moment encoded as a 2D array.
    outcome_cardinalities : List[int]
        List of the cardinalities for the outcomes of the parties.
    skip_party : List[bool]
        Whether each of the parties is considered for normalisation or not.
    """
    eqs = []
    for k, operator in enumerate(moment):
        party = operator[0] - 1
        # Operators that are involved in normalization equalities are
        # those which are unpacked in non-network scenarios
        if (not skip_party[party]
            and operator[-1] == outcome_cardinalities[party] - 2):
            operator_2d = np.expand_dims(operator, axis=0)
            prefix = moment[:k]
            suffix = moment[(k + 1):]
            moments = [moment]
            true_cardinality = outcome_cardinalities[party] - 1
            for outcome in range(true_cardinality - 1):
                variant_operator        = operator_2d.copy()
                variant_operator[0, -1] = outcome
                variant_mon             = np.vstack((prefix,
                                                     variant_operator,
                                                     suffix))
                moments.append(variant_mon)
            if len(moments) == true_cardinality:
                normalization_mon = np.vstack((prefix, suffix))
                eqs.append((normalization_mon, moments))
    return eqs


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
    infSDP : inflation.InflationSDP
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

    relevant_sources = np.flatnonzero(hypergraph[:, party])
    relevant_inflevels = inflevels[relevant_sources]

    assert max_monomial_length <= min(relevant_inflevels), \
        ("You cannot have a longer list of commuting operators" +
         " than the minimum inflation level of said part.")

    # The strategy is building an initial non-negative monomial and apply all
    # inflation symmetries
    initial_monomial = np.zeros(
        (max_monomial_length, 1 + nr_sources + 2), dtype=np.uint16)
    for mon_idx in range(max_monomial_length):
        initial_monomial[mon_idx, 0]    = 1 + party
        initial_monomial[mon_idx, 1:-2] = hypergraph[:, party] * (1 + mon_idx)

    inflation_equivalents = {initial_monomial.tobytes(): initial_monomial}

    all_permutations_per_relevant_source = [
        format_permutations(list(permutations(range(inflevel))))
        for inflevel in relevant_inflevels.flat]
    for permutation in product(*all_permutations_per_relevant_source):
        permuted = initial_monomial.copy()
        for perm_idx, source in enumerate(relevant_sources.flat):
            permuted = mon_lexsorted(apply_source_perm(permuted,
                                                       source,
                                                       permutation[perm_idx]),
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


def make_numerical(symbolic_expressions: Dict[Any, sympy.core.expr.Expr],
                   symbols_to_values: Dict[sympy.core.symbol.Symbol, float]
                   ) -> Dict[Any, float]:
    """Replace the symbols in the values of a dictionary by the corresponding
    numerical values.
    Parameters
    ----------
    symbolic_expressions : Dict[Any, sympy.core.expr.Expr]
        Dictionary where the values are symbolic expressions of some variables.
    symbols_to_values : Dict[sympy.core.symbol.Symbol, float]
        Correspondence of the variables in the expressions and their associated
        numerical values.
    Returns
    -------
    Dict[Any, float]
        The dictionary with samy keys and evaluated expressions as values.
    """
    numeric_values = dict()
    for k, v in symbolic_expressions.items():
        try:
            numeric_values[k] = float(v.evalf(subs=symbols_to_values))
        except AttributeError:
            numeric_values[k] = float(v)
    return numeric_values
