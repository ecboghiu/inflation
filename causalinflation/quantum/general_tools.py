"""
This file contains helper functions to manipulate monomials and generate moment
matrices.
@authors: Alejandro Pozas-Kerstjens, Emanuel-Cristian Boghiu
"""
import copy
import numpy as np
import sympy
from .fast_npa import (apply_source_swap_monomial, factorize_monomial,
                       mon_lessthan_mon, mon_lexsorted,nb_first_index,
                       nb_unique, to_canonical, to_name)
from itertools import permutations, product
from typing import Dict, List, Tuple, Union

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(*args, **kwargs):
        return args[0]

################################################################################
# GENERATION FUNCTIONS                                                         #
################################################################################
def generate_noncommuting_measurements(outs_per_input: List[int],
                                       name: str
                                ) -> List[List[List[sympy.core.symbol.Symbol]]]:
    """Generates the list of sympy.core.symbol.Symbol variables representing the
    measurements for a given party. The variables are treated as non-commuting.
    This code is adapted from ncpol2sdpa. See
    https://github.com/peterwittek/ncpol2sdpa/

    Parameters
    ----------
    outs_per_input : List of int
        The number of outcomes of each measurement for a given party
    name : Str
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


def phys_mon_1_party_of_given_len(hypergraph: np.ndarray,
                                  inflevels: np.array,
                                  party: int,
                                  max_monomial_length: int,
                                  settings_per_party: List[int],
                                  outputs_per_party: List[int],
                                  names: List[str]
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
    hypergraph : np.ndarray
         Hypergraph of the scenario.
    inflevels : np.array
        The number of copies of each source in the inflated scenario.
    party : int
        Party index. NOTE: starting from 0!
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
    List[np.ndarray]
        An array containing all possible positive monomials of the given
        length.
    """
    hypergraph = np.array(hypergraph)
    nr_sources = hypergraph.shape[0]

    assert max_monomial_length <= min(inflevels),                              \
        ("You cannot have a longer list of commuting operators" +
         " than the inflation level.")

    # The initial monomial to which we will apply the symmetries
    # For instance, if max_monomial_length=4, this is something of the form
    # A_1_1_0_xa * A_2_2_0_xa * A_3_3_0_xa * A_4_4_0_xa
    initial_monomial = np.zeros(
        (max_monomial_length, 1+nr_sources+2), dtype=np.uint8)
    for mon_idx in range(max_monomial_length):
        initial_monomial[mon_idx, 0]    = 1 + party
        initial_monomial[mon_idx, -1]   = 0
        initial_monomial[mon_idx, -2]   = 0
        initial_monomial[mon_idx, 1:-2] = hypergraph[:, party] * (1 + mon_idx)

    template_new_mons_aux = [to_name(initial_monomial, names)]
    all_perms_per_source  = [permutations(range(inflevels[source]))
                             for source in range(nr_sources)]
    # Note that we are not applying only the symmetry generators, but all
    # possible symmetries
    for perms in product(*all_perms_per_source):
        permuted = initial_monomial.copy()
        for source in range(nr_sources):
            permuted = apply_source_perm_monomial(
                permuted, source, perms[source], True)
        permuted_name = to_name(permuted, names)
        if permuted_name not in template_new_mons_aux:
            template_new_mons_aux.append(permuted_name)

    template_new_monomials = [np.array(to_numbers(mon, names))
                              for mon in template_new_mons_aux]

    new_monomials = []
    # Insert all combinations of inputs and outputs
    for input_slice in product(*[range(settings_per_party[party])
                                 for _ in range(max_monomial_length)]):
        for output_slice in product(*[range(outputs_per_party[party]-1)
                                      for _ in range(max_monomial_length)]):
            for new_mon_idx in range(len(template_new_monomials)):
                new_monomial = copy.deepcopy(
                    template_new_monomials[new_mon_idx])
                for mon_idx in range(max_monomial_length):
                    new_monomial[mon_idx, -2] = input_slice[mon_idx]
                    new_monomial[mon_idx, -1] = output_slice[mon_idx]
                new_monomials.append(new_monomial)
    return new_monomials


################################################################################
# PROPERTIES AND TRANSFORMATIONS OF MONOMIALS                                  #
################################################################################
def apply_source_perm_monomial(monomial: np.ndarray,
                               source: int,
                               permutation: List,
                               commuting: bool
                               ) -> np.ndarray:
    """Apply a source swap to a monomial.

    Parameters
    ----------
    monomial : np.ndarray
        Input monomial in 2d array format.
    source : int
        The source that is being swapped.
    permutation : List
        The permutation of the copies of the specified source
    commuting : bool
        Whether all the involved operators commute or not.

    Returns
    -------
    np.ndarray
        Input monomial with the specified source swapped.
    """
    new_factors = monomial.copy()
    for i in range(len(monomial)):
        if new_factors[i][1 + source] > 0:
            # Python starts counting at 0
            new_factors[i][1 + source] = permutation[
                new_factors[i][1 + source] - 1] + 1
        else:
            continue
    if commuting:
        return mon_lexsorted(new_factors)
    else:
        return new_factors


def apply_source_permutation_coord_input(columns: List[np.ndarray],
                                         source: int,
                                         permutation: List[int],
                                         commuting: bool
                                         ) -> List[sympy.core.symbol.Symbol]:
    """Apply a specific source permutation to the list of operators used to
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
    commuting : bool
        Whether the operators commute or not.

    Returns
    -------
    List[np.ndarray]
        List of operators with the specified source permuted.
    """
    permuted_op_list = []
    for monomial in columns:
        if np.array_equal(monomial, np.array([0])):
            permuted_op_list.append(monomial)
        else:
            newmon = apply_source_perm_monomial(monomial, source,
                                                np.array(permutation),
                                                commuting)
            canonical = to_canonical(newmon)
            permuted_op_list.append(canonical)
    return permuted_op_list


def factorize_monomials(monomials_as_numbers: np.ndarray,
                        verbose: int = 0
                        ) -> np.ndarray:
    """Applies factorize_momonial to each monomial in the input list
    of monomials.

    Parameters
    ----------
    monomials_as_numbers : np.ndarray
        An ndarray of type object where each row has the integer
        representation of a monomial in the 1st column and in the 2nd
        column a monomial in matrix form (each row is an operator
        and each column has the operator indices).
    verbose : int, optional
        Whether to print progress bar. By default ``0``.

    Returns
    -------
    np.ndarray
        Same as monomials_as_numbers but with the
        factorized monomials.
    """
    monomials_factors = monomials_as_numbers.copy()
    for idx, [_, monomial] in enumerate(tqdm(monomials_factors,
                                        disable=not verbose,
                                        desc="Factorizing monomials        ")):
        monomials_factors[idx][1] = factorize_monomial(monomial)
    return monomials_factors


def is_knowable(monomial: np.ndarray, hypergraph_scenario: np.ndarray) -> bool:
    """Determine whether a given atomic monomial (which cannot be factorized
    into smaller disconnected components) admits an identification with a
    monomial of the original scenario.

    Parameters
    ----------
    monomial : Union[List[List[int]], np.ndarray]
        List of operators, denoted each by a list of indices
    hypergraph_scenario : np.ndarray
        Binary matrix representing the scenario.

    Returns
    -------
    bool
        bool stating whether the monomial is knowable or not
    """
    if type(monomial) == list:
        monomial = np.array(monomial)
    assert monomial.ndim == 2,                                                 \
        ("You must enter a list of monomials. Hence, the number of dimensions "
         + "of the monomial must be 2")
    parties = np.array(monomial)[:, 0].astype(int)
    # If there is more than one monomial of a party, it is not knowable
    if len(set(parties)) != len(parties):
        return False
    else:
        # Indexing of parties begins in 1
        scenario_subhypergraph = hypergraph_scenario[:, parties - 1]
        monomial_sources       = np.array(monomial)[:, 1:-2].T

        # First, test if the monomial corresponds to the scenario.
        monomial_hypergraph = monomial_sources.copy()
        monomial_hypergraph[np.nonzero(monomial_hypergraph)] = 1
        assert all([source in scenario_subhypergraph.tolist()
                    for source in monomial_hypergraph.tolist()]), \
            "The hypergraph corresponding to the monomial does not match a "   \
            + "subgraph of the scenario hypergraph"

        # Check if, for each source, there is at most one copy used
        return all([len(set(source[np.nonzero(source)])) <= 1
                    for source in monomial_sources])


def is_physical(monomial_in: Union[List[List[int]], np.ndarray],
                sandwich_positivity=False
                ) -> bool:
    """Determines whether a monomial is physical, this is, if it always have a
    non-negative expectation value.

    This code also supports the detection of "sandwiches", i.e., monomials
    of the form
    $\\langle \\psi | A_1 A_2 A_1 | \\psi \\rangle$
    where $A_1$ and $A_2$ do not commute. In principle we do not know the
    value of this term. However, note that $A_1$ can be absorbed into
    $| \\psi \\rangle$ forming an unnormalised quantum state
    $| \\psi' \\rangle$, thus
    $\\langle \\psi' | A_2 | \\psi' \\rangle$
    Note that while we know the value $\\langle \\psi | A_2 | \\psi \\rangle$
    we do not know $\\langle \\psi' | A_2 | \\psi' \\rangle$ because of the
    unknown normalisation, however we know it must be positive, thus
    $\\langle \\psi | A_1 A_2 A_1 | \\psi \\rangle \geq 0$.
    This simple example can be extended to various layers of sandwiching.

    Parameters
    ----------
    monomial_in : Union[List[List[int]], np.ndarray]
        Input monomial in 2d array format.
    sandwich_positivity : bool, optional
        Whether to consider sandwiching. By default ``False``.

    Returns
    -------
    bool
        Returns whether the monomial is positive or not.
    """
    monomial = np.array(monomial_in, dtype=np.int8).copy()
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


def label_knowable_and_unknowable(monomials_factors: np.ndarray,
                                  hypergraph: np.ndarray
                                  ) -> np.ndarray:
    """Given the list of monomials factorised, label each monomial as knowable,
    semiknowable or unknowable.

    Parameters
    ----------
    monomials_factors_input : np.ndarray
        Ndarray of factorised monomials. Each row encodes the integer
        representation and the factors of the monomial.
    hypergraph : np.ndarray
        The hypergraph of the network.

    Returns
    -------
    np.ndarray
        Array of the same size as the input, with the labels of each monomial.
    """
    factors_are_knowable       = np.empty_like(monomials_factors)
    factors_are_knowable[:, 0] = monomials_factors[:, 0]
    monomial_is_knowable       = np.empty_like(monomials_factors)
    monomial_is_knowable[:, 0] = monomials_factors[:, 0]
    for idx, [_, factors] in enumerate(monomials_factors):
        factors_known_list = [is_knowable(
            factor, hypergraph) for factor in factors]
        factors_are_knowable[idx][1] = factors_known_list
        if all(factors_known_list):
            knowable = 'Yes'
        elif any(factors_known_list):
            knowable = 'Semi'
        else:
            knowable = 'No'
        monomial_is_knowable[idx][1] = knowable
    return monomial_is_knowable, factors_are_knowable


def remove_sandwich(monomial: np.ndarray) -> np.ndarray:
    """Removes sandwiching/pinching from a monomial. This is, it converts the
    monomial U·A·U^* into A.

    Parameters
    ----------
    monomial : np.ndarray
        Input monomial.

    Returns
    -------
    np.ndarray
        Output monomial without one layer of sandwiching.
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
def as_ordered_factors_for_powers(monomial: sympy.core.symbol.Symbol
                                  ) -> List[sympy.core.symbol.Symbol]:
    """If we have powers of a monomial, such as A**3, return a list with
    the factors, [A, A, A]. This is for treating cases where we need the
    individual factors of a monomial but we can not extract them with
    .as_ordered_factors()

    Parameters
    ----------
    monomial : sympy.core.symbol.Symbol
        Symbolic monomial, possible with powers.

    Returns
    -------
    List[sympy.core.symbol.Symbol]
        List of all the symbolic factors, with the powers expanded.
    """
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


def cols_num2sym(ordered_cols_coord: List[List[List[int]]],
                  names: str,
                  n_sources: int,
                  measurements: List[List[List[sympy.core.symbol.Symbol]]]
                  ) -> List[sympy.core.symbol.Symbol]:
    """Go from the output of build_columns to a list of symbolic operators

    Parameters
    ----------
    ordered_cols_coord : List[List[List[int]]]
        Generating set as a list of monomials represented as an array.
    names : str
        Names of each party.
    n_sources : int
        Number of sources.
    measurements : List[List[List[sympy.core.symbol.Symbol]]]
        List of symbolic operators representing the measurements. The list is
        nested such that the first index corresponds to the party, the
        second index to the measurement, and the third index to the outcome.

    Returns
    -------
    List[sympy.core.symbol.Symbol]
        The generating set but with symbolic monomials.
    """
    flatmeas = np.array(flatten(measurements))
    measnames = np.array([str(meas) for meas in flatmeas])

    res = [None] * len(ordered_cols_coord)
    for ii, elements in enumerate(ordered_cols_coord):
        if np.array_equal(elements, np.array([0])):
            res[ii] = sympy.S.One
        else:
            product = sympy.S.One
            for element in elements:
                party = element[0]
                name = names[party - 1] + '_'
                for s in element[1:1 + n_sources]:
                    name += str(s) + '_'
                name += str(element[-2]) + '_' + str(element[-1])
                term = flatmeas[measnames == name][0]
                product *= term
            res[ii] = product
    return res


def from_numbers_to_flat_tuples(lst: List[List[int]]) -> List[Tuple[int]]:
    """Flatten all monomials in the list represented as lists of lists to a
    flat tuple.

    This is useful for dictionaries, as list of lists are not hashable.

    Parameters
    ----------
    list : List[List[int]]
        List of monomials encoded as lists of lists (or array of arrays).

    Returns
    -------
    List[Tuple[int]]
        List of monomials encoded as flat tuples of integers.
    """
    tuples = []
    for element in lst:
        if np.array_equal(element, np.array([0])):
            tuples.append(tuple([0]))
        else:
            tuples.append(tuple(flatten(element.tolist())))
    return tuples


def monomialset_name2num(monomials: np.ndarray, names: List[str]) -> np.ndarray:
    """Change each monomial in the list of monomials from string to matrix
    representation.

    Parameters
    ----------
    monomials : np.ndarray
        Input monomials list.
    names : List[str]
         List of party names

    Returns
    -------
    np.ndarray
        Same dimensions as input monomials, but each monomial is in
        string format.
    """
    monomials_numbers = monomials.copy()
    for i, line in enumerate(tqdm(monomials,
                                 disable=True,
                                 desc="Converting monomial names to numbers")):
        monomials_numbers[i][1] = to_numbers(line[1], names)
    monomials_numbers[:, 0] = monomials_numbers[:, 0].astype(int)
    return monomials_numbers


def monomialset_num2name(monomials_factors: np.ndarray,
                         names: List[str]
                         ) -> np.ndarray:
    """Change the list of monomials from a list of arrays to a list of strings.

    Parameters
    ----------
    monomials_factors : np.ndarray
        List of monomials.
    names : List[str]
        names[i] is the name of party i+1 (parties in [1,2,3,4...]).

    Returns
    -------
    np.ndarray
        Returns the input with the monomials replaced by their
        string representation.
    """
    monomials_factors_names = monomials_factors.copy()
    for idx, [_, monomial_factors] in enumerate(monomials_factors_names):
        factors_names_list = [to_name(factors, names)
                              for factors in monomial_factors]
        monomials_factors_names[idx][1] = factors_names_list
    return monomials_factors_names


def string2prob(term: str, max_nr_of_parties: int) -> sympy.core.symbol.Symbol:
    """Converts a string to a symbolic probability with the correct indices.
    For example 'A_0_1_0*B_0_2_3' is converted to pAB(03|12).

    Parameters
    ----------
    term : _type_
        Input monomial as a string.
    max_nr_of_parties : _type_
        The number of terms in the monomial.

    Returns
    -------
    sympy.core.symbol.Symbol
        The symbolic probability, e.g., p(00|01).
    """
    if term == '1':
        return 1
    elif term == '0':
        return 0
    factors = term.split('*')
    factors = [list(factor.split('_')) for factor in factors]
    factors = np.array(factors)
    parties = factors[:, 0]
    inputs  = factors[:, -2]
    outputs = factors[:, -1]
    name = 'p'
    # Add parties if we are marginalizing over a distribution
    if len(parties) < max_nr_of_parties:
        for p in parties:
            name += p
    name += '('
    for o in outputs:
        name += o
    name += '|'
    for i in inputs:
        name += i
    name += ')'
    return sympy.symbols(name, commuting=True)


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
        Monomial in list of lists format (equivalent to 2d array format by
        calling np.array() on the result).
    """
    parties_names_dict = {name: i + 1 for i, name in enumerate(parties_names)}

    if isinstance(monomial, str):
        monomial_parts = monomial.split('*')
    else:
        factors = as_ordered_factors_for_powers(monomial)
        monomial_parts = [str(factor) for factor in factors]

    monomial_parts_indices = []
    for part in monomial_parts:
        atoms = part.split('_')
        indices = ([parties_names_dict[atoms[0]]]
                   + [int(j) for j in atoms[1:-2]]
                   + [int(atoms[-2]), int(atoms[-1])])
        monomial_parts_indices.append(indices)
    return monomial_parts_indices


def to_representative_aux(monomial_component: np.ndarray
                          ) -> np.ndarray:
    """Auxiliary function for to_representative. It applies source swaps
    until we reach a stable point in terms of lexiographic ordering. This might
    not be a global optimum if we also take into account the commutativity.

    Parameters
    ----------
    monomial_component : np.ndarray
        Input monomial.

    Returns
    -------
    np.ndarray
        An equivalent monomial closer to its representative form.
    """
    monomial_component = to_canonical(monomial_component)
    new_mon            = monomial_component.copy()
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


def to_representative(mon: np.ndarray,
                      inflevels: np.array,
                      commuting: bool
                      ) -> np.ndarray:
    """Take a monomial and applies inflation symmetries to bring it to a
    canonical form.

    Example:
    Assume the monomial is something like:
    < D^350_00 D^450_00 D^150_00 E^401_00 F^031_00 >

    Let us put the inflation copies as a matrix:

    [[3 5 0],
     [4 5 0],
     [1 5 0],
     [4 0 1],
     [0 3 1]]

    For each column we assign to the first row index 1. Then the next different
    one will be 2, and the third different will be 3.
    We ignore 0s.
    Col1 (unique=[3,4,1])) [3 4 1 4 0] --> [1 4 3 4 0] --> [1 2 3 2 0] Done!
    Col2 (unique=[5,3])    [5 5 5 0 3] --> [1 1 1 0 3] --> [1 1 1 0 2] Done!

    Parameters
    ----------
    mon : np.ndarray
        Input monomial that cannot be further factorised.
    inflevels : np.array
        Number of copies of each source in the inflated graph.
    commuting : bool
        Whether all the involved operators commute or not.

    Returns
    -------
    np.ndarray
        Input monomial in canonical form.

    """
    mon_aux = to_representative_aux(mon)

    # The application of symmetries plus  applying commutation rules can give
    # an even smaller monomial (lexicographically). To deal with this, we apply
    # all possible source swaps and then apply commutation rules, and if the
    # resulting monomial is smaller, we accept it.
    nr_sources = inflevels.shape[0]
    all_perms_per_source = [permutations(
                      range(inflevels[source])) for source in range(nr_sources)]

    final_monomial = mon_aux.copy()
    prev = mon_aux
    while True:
        for perms in product(*all_perms_per_source):
            permuted = final_monomial.copy()
            for source in range(nr_sources):
                permuted = apply_source_perm_monomial(
                    permuted, source, perms[source], commuting)
            permuted = to_canonical(permuted)
            if mon_lessthan_mon(permuted, final_monomial):
                final_monomial = permuted
        if np.array_equal(final_monomial, prev):
            break
        prev = final_monomial

    return final_monomial

################################################################################
# OTHER FUNCTIONS                                                              #
################################################################################
def clean_coefficients(cert: Dict[int, float],
                       chop_tol: float=1e-10,
                       round_decimals: int=3) -> np.array:
    """Clean the list of coefficients in a certificate.

    Parameters
    ----------
    coefficients : numpy.array
      The list of coefficients.
    chop_tol : float, optional
      Coefficients in the dual certificate smaller in absolute value are
      set to zero. Defaults to 1e-10.
    round_decimals : int, optional
      Coefficients that are not set to zero are rounded to the number
      of decimals specified. Defaults to 3.

    Returns
    -------
    numpy.array
      The cleaned-up coefficients.
    """
    max_val = np.abs(np.max(list(cert.values())))
    for key, val in cert.items():
        if abs(val) < chop_tol:
            val = 0
        cert[key] = np.round(val / max_val, decimals=round_decimals)
    return cert


def compute_numeric_value(mon_string: str,
                          p_array: np.ndarray,
                          parties_names: List[str]
                          ) -> float:
    """Given a monomial and a probability distribution p_array called as
    p_array[a,b,c,...,x,y,z,...], return the numerical value of the probability
    associated to the monomial.

    Note that this accepts marginals, for example, p(a|x), and then it
    automatically computes all the summations over p[a,b,c,...,x,y,z,...].

    Parameters
    ----------
    mon_str : String
        Monomial associated to the probability.
    p_array : np.ndarray
        The probability distribution of dims
        (outcomes_per_party, settings_per_party).
    parties_names : List[str]
        List of party names.

    Returns
    -------
    float
        The value of the symbolic probability (which can be a marginal)

    Examples
    --------
    >>> p = 'A_1_2_3_i_o'
    >>> compute_numeric_value(p, [2,2], [2,2], parray)
    parray[o,:,i,0].sum()

    Note that we take the first setting (=0) for marginalised parties, in the
    example above, the second party is marginalised.
    """
    n_parties  = p_array.ndim // 2
    components = np.array([factor.split('_')
                           for factor in mon_string.split('*')])
    names   = components[:,  0]
    inputs  = components[:, -2].astype(int)
    outputs = components[:, -1].astype(int).tolist()
    dont_marginalize = [parties_names.index(name) for name in names]
    indices_to_sum   = list(set(range(n_parties)) - set(dont_marginalize))
    marginal_dist    = np.sum(p_array, axis=tuple(indices_to_sum))

    input_list       = np.zeros(n_parties, dtype=int)
    input_list[dont_marginalize] = inputs
    inputs_outputs   = outputs + input_list.tolist()
    return marginal_dist[tuple(inputs_outputs)]


def find_permutation(list1: List, list2: List) -> List[int]:
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
        original_dict = {element: num for element, num in zip(list1,
                                                            range(len(list1)))}
        return [original_dict[element] for element in list2]


def flatten(nested):
    """Keeps flattening a nested lists of lists until the
    first element of the resulting list is not a list.

    Parameters
    ----------
    nested : List of lists of arbitrary depth.

    Returns
    -------
    list
        The flat list of elements
    """
    while type(nested[0]) == list:
        nested = [item for sublist in nested for item in sublist]
    return nested
