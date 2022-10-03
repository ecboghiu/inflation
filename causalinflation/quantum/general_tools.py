"""
This file contains helper functions to manipulate monomials and generate moment
matrices.
@authors: Alejandro Pozas-Kerstjens, Emanuel-Cristian Boghiu
"""
from copy import deepcopy
from itertools import chain, permutations, product
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import sympy

from .fast_npa import (factorize_monomial,
                       mon_lexsorted,
                       to_name,
                       hasty_to_canonical,
                       apply_source_permplus_monomial)

try:
    from numba import jit
    from numba.types import bool_
except ImportError:
    def jit(*args, **kwargs):
        return lambda f: f


    bool_ = bool

try:
    from tqdm import tqdm
except ImportError:
    from ..utils import blank_tqdm as tqdm

nopython = False
cache = False


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
    all_permutationsplus_per_source = [
        increase_values_by_one_and_prepend_with_column_of_zeros(list(permutations(range(inflevel))))
        for inflevel in inflevels.flat]
    # Note that we are not applying only the symmetry generators, but all
    # possible symmetries
    for perms_plus in product(*all_permutationsplus_per_source):
        permuted = initial_monomial.copy()
        for source in range(nr_sources):
            permuted = mon_lexsorted(apply_source_permplus_monomial(
                monomial=permuted,
                source=source,
                permutation_plus=perms_plus[source]), lexorder)
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
                new_monomial = deepcopy(template_new_monomials[new_mon_idx])
                for mon_idx in range(max_monomial_length):
                    new_monomial[mon_idx, -2] = input_slice[mon_idx]
                    new_monomial[mon_idx, -1] = output_slice[mon_idx]
                new_monomials.append(new_monomial)
    return new_monomials


def increase_values_by_one_and_prepend_with_column_of_zeros(array) -> np.ndarray:
    array_plus = np.asarray(array) + 1
    padding = np.zeros((len(array_plus), 1), dtype=int)
    return np.hstack((padding, array_plus))


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
        symbols.append(sympy.Symbol("_".join([names[letter[0] - 1]] +
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
        monomial_parts = monomial.split("*")
    else:
        factors = flatten_symbolic_powers(monomial)
        monomial_parts = [str(factor) for factor in factors]

    monomial_parts_indices = []
    for part in monomial_parts:
        atoms = part.split("_")
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
        monomial_parts = monomial.split("*")
    else:
        factors = flatten_symbolic_powers(monomial)
        monomial_parts = [str(factor) for factor in factors]

    monomial_parts_indices = []
    for part in monomial_parts:
        atoms = part.split("_")
        indices = ([parties_names_dict[atoms[0]]]
                   + [int(j) for j in atoms[1:-2]]
                   + [int(atoms[-2]), int(atoms[-1])])
        monomial_parts_indices.append(indices)
    return monomial_parts_indices


def to_repr_lower_copy_indices_with_swaps(monomial_component: np.ndarray) -> np.ndarray:
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
    new_mon = monomial_component.copy()
    nr_sources = monomial_component.shape[1] - 3
    # We temporarily pad the monomial with a row a zeroes at the front.
    new_mon_padded_transposed = np.concatenate((np.zeros_like(new_mon[:1]), new_mon)).T
    for source in range(nr_sources):
        source_inf_copy_nrs = new_mon_padded_transposed[1 + source]
        # Numpy's return_inverse helps us reset the indices.
        _, unique_positions = np.unique(source_inf_copy_nrs, return_inverse=True)
        new_mon_padded_transposed[1 + source] = unique_positions
    return new_mon_padded_transposed.T[1:]


def clean_coefficients(cert_dict: Dict[str, float],
                       chop_tol: float = 1e-10,
                       round_decimals: int = 3) -> Dict:
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
    processed_cert = deepcopy(cert_dict)
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


def flatten(nested):
    """Keeps flattening a nested lists of lists until  the
    first element of the resulting list is not a list.
    """
    if isinstance(nested, np.ndarray):
        return nested.ravel().tolist()
    else:
        while isinstance(nested[0], Iterable):
            nested = list(chain.from_iterable(nested))
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
                sympy.Symbol(name + "_" + str(x) + "_" + str(o),
                             commutative=False)
            )
        ops_per_input.append(ops_per_output_per_input)
    return ops_per_input
