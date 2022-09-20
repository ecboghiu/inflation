"""
This file contains helper functions to manipulate monomials and generate moment
matrices.
@authors: Alejandro Pozas-Kerstjens, Emanuel-Cristian Boghiu
"""
import copy
import itertools
from functools import lru_cache
from itertools import permutations, product

import numpy as np
import sympy

from typing import Any, Dict, Iterable, List, Tuple, Union
from .fast_npa import (apply_source_swap_monomial,
                       factorize_monomial,
                       mon_lessthan_mon,
                       mon_lexsorted,
                       nb_unique,
                       to_canonical,
                       to_name,
                       mon_equal_mon,
                       reverse_mon)

try:
    from numba import jit
    from numba import int64 as int64_
    from numba.types import bool_
except ImportError:
    def jit(*args, **kwargs):
        return lambda f: f


    int64_ = np.int64
    bool_ = bool

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(*args, **kwargs):
        return args[0]

nopython = False
cache = False


def find_permutation(list1: List,
                     list2: List
                     ) -> List:
    """Returns the permutation that transforms list2 in list1.

    Parameters
    ----------
    list1 : List
        First input list.
    list2 : List
        Second input list.

    Returns
    -------
    List
        Permutation the brings list2 to list1.

    Raises
    ------
    Exception
        If the lengths are different, or if the elements of the two lists are
        different.
    """
    if (len(list1) != len(list2)) or (set(list1) != set(list2)):
        raise Exception('The two lists are not permutations of one another')
    else:
        original_dict = {x: i for i, x in enumerate(list1)}
        return [original_dict[x] for x in list2]


@jit(nopython=nopython)
def apply_source_permplus_monomial(monomial: np.ndarray,
                                   source: int,
                                   permutation_plus: np.ndarray,
                                   commuting: bool_,
                                   lexorder
                                   ) -> np.ndarray:
    """This applies a source swap to a monomial.

    We assume in the monomial that all operators COMMUTE with each other.

    Parameters
    ----------
    monomial : numpy.ndarray
        Input monomial in 2d array format.
    source : int
        The source that is being swapped.
    permutation_plus : numpy.ndarray
        The permutation of the copies of the specified source.
        The format for the permutation here is to use indexing starting at one, so the permutation must be
        padded with a leading zero.
    commuting : bool
        Whether all the involved operators commute or not.

    Returns
    -------
    np.ndarray
        Input monomial with the specified source swapped.
    """
    new_factors = monomial.copy()
    new_factors[:, 1 + source] = np.take(permutation_plus, new_factors[:, 1 + source])
    if commuting:
        return mon_lexsorted(new_factors, lexorder)
    else:
        return new_factors


def apply_source_permutation_coord_input(columns: List[np.ndarray],
                                         source: int,
                                         permutation: Union[np.ndarray, Tuple[int]],
                                         commuting: bool,
                                         notcomm,
                                         lexorder
                                         ) -> List[np.ndarray]:
    """Applies a specific source permutation to the list of operators used to
    define the moment matrix. Outputs the permuted list of operators.
    The operators are enconded as lists of numbers denoting
    [party, source_1_copy, source_2_copy, ..., input, output]
    A product of operators is a list of such lists transformed into a
    np.ndarray.

    Parameters
    ----------
    columns : List[np.ndarray]
        Generating set as a list of monomials in 2d array format.
    source : int
        Source that is being swapped.
    permutation : List[int]
        Permutation of the copies of the specified source.
        The format for the permutation here is to use indexing starting at one, so the permutation must be
        padded with a leading zero.
    commuting : bool
        Whether the operators commute or not.


    Returns
    -------
    List[np.ndarray]
        List of operators with the specified source permuted.

    """
    permuted_op_list = []
    for monomial in columns:
        (row_count, col_count) = monomial.shape
        if row_count == 0 or col_count == 1:
            permuted_op_list.append(monomial)
        else:
            newmon = apply_source_permplus_monomial(monomial, source,
                                                np.asarray(permutation),
                                                commuting, lexorder)
            canonical = to_canonical(newmon, notcomm, lexorder)
            permuted_op_list.append(canonical)
    return permuted_op_list




def phys_mon_1_party_of_given_len(hypergraph: np.ndarray,
                                  inflevels: np.ndarray,
                                  party: int,
                                  max_monomial_length: int,
                                  settings_per_party: Tuple[int],
                                  outputs_per_party: Tuple[int],
                                  names: Tuple[str],
                                  lexorder
                                  ) -> List[np.ndarray]:
    """Generate all possible positive monomials given a scenario and a maximum
    length. Note that the maximum length cannot be greater than the minimum
    number of copies for each source that the party has access to. For example,
    if party 2 has access to 3 sources, the first has 3 copies, the second
    4 copies and the third 5 copies, the maximum length cannot be greater than
    3. This is because the extra operators will not commute with the ones before
    as they will be sharing support.

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
    names : List[str]
        names[i] is the string name of the party i+1.

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

    # The initial monomial to which we will apply the symmetries
    # For instance, if max_monomial_length=4, this is something of the form
    # A_1_1_0_xa * A_2_2_0_xa * A_3_3_0_xa * A_4_4_0_xa
    initial_monomial = np.zeros(
        (max_monomial_length, 1 + nr_sources + 2), dtype=np.uint16)
    for mon_idx in range(max_monomial_length):
        initial_monomial[mon_idx, 0] = 1 + party
        initial_monomial[mon_idx, -1] = 0
        initial_monomial[mon_idx, -2] = 0
        initial_monomial[mon_idx, 1:-2] = hypergraph[:, party] * (1 + mon_idx)

    template_new_mons_aux = [to_name(initial_monomial, names)]
    all_perms_per_source = [np.array(list(permutations(range(inflevels[source]))), dtype=int)
                            for source in range(nr_sources)]
    # Note that we are not applying only the symmetry generators, but all
    # possible symmetries
    all_permutationsplut_per_source = []
    for array_of_perms in all_perms_per_source:
        permutations_plus = array_of_perms + 1
        padding = np.zeros((len(permutations_plus), 1), dtype=int)
        permutations_plus = np.hstack((padding, permutations_plus))
        all_permutationsplut_per_source.append(permutations_plus)
    del all_perms_per_source
    for perms_plus in product(*all_permutationsplut_per_source):
        permuted = initial_monomial.copy()
        for source in range(nr_sources):
            permuted = apply_source_permplus_monomial(
                monomial=permuted,
                source=source,
                permutation=perms_plus[source],
                commuting=True,
                lexorder=lexorder)
        permuted_name = to_name(permuted, names)
        if permuted_name not in template_new_mons_aux:
            template_new_mons_aux.append(permuted_name)

    template_new_monomials = [
        np.asarray(to_numbers(mon, names)) for mon in template_new_mons_aux]

    new_monomials = []
    # Insert all combinations of inputs and outputs
    for input_slice in product(*[range(settings_per_party[party])
                                 for _ in range(max_monomial_length)]):
        for output_slice in product(*[range(outputs_per_party[party] - 1)
                                      for _ in range(max_monomial_length)]):
            for new_mon_idx in range(len(template_new_monomials)):
                new_monomial = copy.deepcopy(
                    template_new_monomials[new_mon_idx])
                for mon_idx in range(max_monomial_length):
                    new_monomial[mon_idx, -2] = input_slice[mon_idx]
                    new_monomial[mon_idx, -1] = output_slice[mon_idx]
                new_monomials.append(new_monomial)
    return new_monomials


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
    # This is for treating cases like A**2, where we want factors = [A, A]
    # and this behaviour doesn't work with .as_ordered_factors()
    factors = monomial.as_ordered_factors()
    factors_expanded = []
    for f_temp in factors:
        base, exp = f_temp.as_base_exp()
        if exp == 1:
            factors_expanded.append(base)
        elif exp > 1:
            for _ in range(exp):
                factors_expanded.append(base)
    factors = factors_expanded
    return factors


def to_symbol(monomial: np.ndarray, names: Tuple[str]) -> sympy.core.symbol.Symbol:
    """Converts a monomial to a symbolic representation.
    """
    if np.array_equal(monomial, np.array([[0]], dtype=np.uint16)):
        return sympy.S.One

    if type(monomial) == str:
        monomial = to_numbers(monomial, names)
    monomial = monomial.tolist()
    symbols = []
    for letter in monomial:
        symbols.append(sympy.Symbol('_'.join([names[letter[0] - 1]] +
                                             [str(i) for i in letter[1:]]),
                                    commutative=False))
    prod = sympy.S.One
    for s in symbols:
        prod *= s
    return prod


def to_numbers(monomial: str,
               parties_names: Tuple[str]
               ) -> np.ndarray:
    """Monomial from string to matrix representation.

    Given a monomial input in string format, return the matrix representation
    where each row represents an operators and the columns are operator labels
    such as party, inflation copies and input and output cardinalities.

    Parameters
    ----------
    monomial : str
        Monomial in string format.
    parties_names : Tuple[str]
        Tuple of party names.

    Returns
    -------
    Tuple[Tuple[int]]
        Monomial in tuple of tuples format (equivalent to 2d array format by
        calling np.array() on the result).
    """
    parties_names_dict = {name: i + 1 for i, name in enumerate(parties_names)}

    if isinstance(monomial, str):
        monomial_parts = monomial.split('*')
    else:
        factors = flatten_symbolic_powers(monomial)
        monomial_parts = [str(factor) for factor in factors]

    monomial_parts_indices = []
    for part in monomial_parts:
        atoms = part.split('_')
        if atoms[0] not in parties_names_dict.keys():
            raise Exception(f"Party name {atoms[0]} not recognized.")
        indices = ((parties_names_dict[atoms[0]],)
                   + tuple(int(j) for j in atoms[1:-2])
                   + (int(atoms[-2]), int(atoms[-1])))
        monomial_parts_indices.append(indices)

    return np.array(monomial_parts_indices, dtype=np.uint16)


def is_knowable(monomial: np.ndarray) -> bool:
    """Determine whether a given atomic monomial (which cannot be factorized
    into smaller disconnected components) admits an identification with a
    monomial of the original scenario.

    Parameters
    ----------
    monomial : np.ndarray
        List of operators, denoted each by a list of indices

    Returns
    -------
    bool
        Whether the monomial is knowable or not.
    """
    assert monomial.ndim == 2, "You must enter a list of monomials. Hence," \
                               + " the number of dimensions of monomial must be 2"
    parties = monomial[:, 0].astype(int)
    # If there is more than one monomial of a party, it is not knowable
    if len(set(parties)) != len(parties):
        return False
    else:
        # We see if, for each source, there is at most one copy used
        return all([len(set(source[np.nonzero(source)])) <= 1
                    for source in monomial[:, 1:-2].T])


def is_physical(monomial_in: Iterable[Iterable[int]],
                sandwich_positivity=True
                ) -> bool:
    """Determines whether a monomial is physical, this is, if it always have a
    non-negative expectation value.

    This code also supports the detection of "sandwiches", i.e., monomials
    of the form :math:`\\langle \\psi | A_1 A_2 A_1 | \\psi \\rangle` where
    :math:`A_1` and :math:`A_2` do not commute. In principle we do not know the
    value of this term. However, note that :math:`A_1` can be absorbed into
    :math:`| \\psi \\rangle` forming an unnormalised quantum state
    :math:`| \\psi' \\rangle`, thus :math:`\\langle \\psi'|A_2|\\psi' \\rangle`.
    Note that while we know the value :math:`\\langle\\psi |A_2| \\psi\\rangle`,
    we do not know :math:`\\langle \\psi' | A_2 | \\psi' \\rangle` because of
    the unknown normalisation, however we know it must be non-negative,
    :math:`\\langle \\psi | A_1 A_2 A_1 | \\psi \\rangle \geq 0`.
    This simple example can be extended to various layers of sandwiching.

    Parameters
    ----------
    monomial_in : Union[List[List[int]], numpy.ndarray]
        Input monomial in 2d array format.
    sandwich_positivity : bool, optional
        Whether to consider sandwiching. By default ``False``.

    Returns
    -------
    bool
        Whether the monomial is positive or not.
    """
    if not len(monomial_in):
        return monomial_in
    monomial = np.array(monomial_in, dtype=np.uint16, copy=True)
    if sandwich_positivity:
        monomial = remove_sandwich(monomial)
    res = True
    parties = np.unique(monomial[:, 0])
    for party in parties:
        party_monomial = monomial[monomial[:, 0] == party]
        if not len(party_monomial) == 1:
            factors = factorize_monomial(party_monomial)
            if len(factors) != len(party_monomial):
                res *= False
                break
    return res


def remove_sandwich(monomial: np.ndarray) -> np.ndarray:
    """Removes sandwiching/pinching from a monomial. This is, it converts the
    monomial represented by :math:`U A U^\dagger` into :math:`A`.

    Parameters
    ----------
    monomial : numpy.ndarray
        Input monomial.

    Returns
    -------
    numpy.ndarray
        The monomial without one layer of sandwiching.
    """
    new_monomial = np.empty((0, monomial[0, :].shape[0]), dtype=int)
    parties = np.unique(monomial[:, 0])
    for party in parties:
        party_monomial = monomial[monomial[:, 0] == party].copy()
        party_monomial_factorized = factorize_monomial(party_monomial)
        for factor in party_monomial_factorized:
            factor_copy = factor
            if len(factor) > 1:
                if np.array_equal(factor[0], factor[-1]):
                    factor_copy = np.delete(factor, (0, -1), axis=0)
            new_monomial = np.append(new_monomial, factor_copy, axis=0)
    return new_monomial


################################################################################
# REPRESENTATIONS AND CONVERSIONS                                              #
################################################################################

def to_numbers(monomial: str, parties_names: List[str]) -> List[List[int]]:
    """Convert monomial from string to matrix representation.

    Given a monomial input in string format, return the matrix representation
    where each row represents an operators and the columns are operator labels
    such as party, inflation copies and input and output cardinalities.

    Parameters
    ----------
    monomial : str
        Monomial in string format.
    parties_names : List[str]
        List of party names.

    Returns
    -------
    List[List[int]]
        The monomial in list of lists format (equivalent to 2d array format by
        calling np.array() on the result).
    """
    parties_names_dict = {name: i + 1 for i, name in enumerate(parties_names)}

    if isinstance(monomial, str):
        monomial_parts = monomial.split('*')
    else:
        factors = flatten_symbolic_powers(monomial)
        monomial_parts = [str(factor) for factor in factors]

    monomial_parts_indices = []
    for part in monomial_parts:
        atoms = part.split('_')
        indices = ([parties_names_dict[atoms[0]]]
                   + [int(j) for j in atoms[1:-2]]
                   + [int(atoms[-2]), int(atoms[-1])])
        monomial_parts_indices.append(indices)
    return monomial_parts_indices


def to_repr_lower_copy_indices_with_swaps(monomial_component: np.ndarray,
                                          notcomm: np.ndarray,
                                          lexorder: np.ndarray) -> np.ndarray:
    """Auxiliary function for to_representative. It applies source swaps
    until we reach a stable point in terms of lexiographic ordering. This might
    not be a global optimum if we also take into account the commutativity.

    Parameters
    ----------
    monomial_component : numpy.ndarray
        Input monomial.

    Returns
    -------
    numpy.ndarray
        An equivalent monomial closer to its representative form.
    """
    monomial_component = to_canonical(
        np.asarray(monomial_component), notcomm, lexorder)
    new_mon = monomial_component.copy()
    for source in range(monomial_component.shape[1] - 3):
        source_inf_copy_nrs = monomial_component[:, 1 + source]
        # This returns the unique values unsorted
        uniquevals, _ = nb_unique(source_inf_copy_nrs)
        uniquevals = uniquevals[uniquevals > 0]  # Remove the 0s
        for idx, old_val in enumerate(uniquevals):
            new_val = idx + 1
            if old_val > new_val:
                new_mon = apply_source_swap_monomial(new_mon, source,
                                                     old_val, new_val)
    return new_mon


def to_repr_swap_plus_commutation(mon_aux: np.ndarray,
                                  inflevels: np.ndarray,
                                  notcomm: np.ndarray,
                                  lexorder: np.ndarray,
                                  commuting: bool) -> np.ndarray:
    nr_sources = inflevels.shape[0]
    all_perms_per_source = [np.array(list(permutations(range(inflevels[source]))), dtype=int)
                            for source in range(nr_sources)]
    # Note that we are not applying only the symmetry generators, but all
    # possible symmetries
    all_permutationsplut_per_source = []
    for array_of_perms in all_perms_per_source:
        permutations_plus = array_of_perms + 1
        padding = np.zeros((len(permutations_plus), 1), dtype=int)
        permutations_plus = np.hstack((padding, permutations_plus))
        all_permutationsplut_per_source.append(permutations_plus)
    del all_perms_per_source
    final_monomial = mon_aux.copy()
    prev = mon_aux
    while True:
        for perms_plus in product(*all_permutationsplut_per_source):
            permuted = final_monomial.copy()
            for source in range(nr_sources):
                permuted = apply_source_permplus_monomial(
                    monomial=permuted,
                    source=source,
                    permutation_plus=perms_plus[source],
                    commuting=commuting,
                    lexorder=lexorder)
            permuted = to_canonical(permuted, notcomm, lexorder)
            if mon_lessthan_mon(permuted, final_monomial, lexorder):
                final_monomial = permuted
        if np.array_equal(final_monomial, prev):
            break
        prev = final_monomial

    return final_monomial


def to_representative(mon: np.ndarray,
                      inflevels: np.ndarray,
                      notcomm: np.ndarray,
                      lexorder: np.ndarray,
                      commuting: bool_ = False,
                      consider_conjugation_symmetries: bool_ = True,
                      swaps_plus_commutations: bool_ = True,
                      ) -> np.ndarray:
    """Take a monomial and applies inflation symmetries to bring it to a
    canonical form.

    Example: Assume the monomial is :math:`\\langle D^{350}_{00} D^{450}_{00}
    D^{150}_{00} E^{401}_{00} F^{031}_{00} \\rangle`. Let us put the inflation
    copies as a matrix:

    ::

        [[3 5 0],
         [4 5 0],
         [1 5 0],
         [4 0 1],
         [0 3 1]]

    For each column we assign to the first row index 1. Then the next different
    one will be 2, and so on. Therefore, the representative of the monomial
    above is :math:`\\langle D^{110}_{00} D^{210}_{00} D^{350}_{00} E^{201}_{00}
    F^{021}_{00} \\rangle`.

    Parameters
    ----------
    mon : numpy.ndarray
        Input monomial that cannot be further factorised.
    inflevels : np.ndarray
        Number of copies of each source in the inflated graph.
    commuting : bool
        Whether all the involved operators commute or not.

    Returns
    -------
    numpy.ndarray
        The canonical form of the input monomial.
    """
    if mon_equal_mon(mon, np.array([[0]], dtype=np.uint16)):
        return mon

    # We apply source swaps until we reach a stable point in terms of
    # lexiographic ordering.
    final_monomial = to_repr_lower_copy_indices_with_swaps(mon, notcomm, lexorder)

    # Before we didn't consider that applying a source swap that decreases the
    # ordering following by applying commutation rules can give us a smaller
    # monomial lexicographically.
    if swaps_plus_commutations:
        final_monomial = to_repr_swap_plus_commutation(final_monomial, inflevels,
                                                       notcomm,
                                                       lexorder, commuting)

    if consider_conjugation_symmetries:
        mon_dagger = reverse_mon(mon)
        mon_dagger_aux = to_repr_lower_copy_indices_with_swaps(mon_dagger, notcomm, lexorder)
        if swaps_plus_commutations:
            mon_dagger_aux = to_repr_swap_plus_commutation(mon_dagger_aux,
                                                           inflevels,
                                                           notcomm,
                                                           lexorder,
                                                           commuting)
        if mon_lessthan_mon(mon_dagger_aux, final_monomial, lexorder):
            final_monomial = mon_dagger_aux

    return final_monomial


def clean_coefficients(cert_dict: Dict[str, float],
                       chop_tol: float = 1e-10,
                       round_decimals: int = 3) -> np.ndarray:
    """Clean the list of coefficients in a certificate.

    Parameters
    ----------
    cert_dict : Dict[str, float]
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
    processed_cert = copy.deepcopy(cert_dict)
    vars   = np.asarray(list(processed_cert.keys()))
    coeffs = np.asarray(list(processed_cert.values()))
    # Take the biggest one and make it 1
    normalising_factor = np.max(np.abs(coeffs[np.abs(coeffs) > chop_tol]))
    coeffs /= normalising_factor
    # Set to zero very small coefficients
    coeffs[np.abs(coeffs) <= chop_tol] = 0
    # Round
    coeffs = np.round(coeffs, decimals=round_decimals)
    return dict(zip(vars, coeffs))


def flatten(nested):
    """Keeps flattening a nested lists of lists until  the
    first element of the resulting list is not a list.
    """
    if isinstance(nested, np.ndarray):
        return nested.ravel().tolist()
    else:
        while isinstance(nested[0], Iterable):
            nested = list(itertools.chain.from_iterable(nested))
        return nested


def generate_operators(outs_per_input: List[int],
                       name: str
                       ) -> List[List[List[sympy.core.symbol.Symbol]]]:
    """Generates the list of ``sympy.core.symbol.Symbol`` variables representing
    the measurements for a given party. The variables are treated as
    non-commuting. This code is adapted from `ncpol2sdpa
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
                sympy.Symbol(name + '_' + str(x) + '_' + str(o),
                             commutative=False)
            )
        ops_per_input.append(ops_per_output_per_input)
    return ops_per_input
