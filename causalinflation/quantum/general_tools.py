"""
This file contains helper functions to manipulate monomials and generate moment
matrices.
@authors: Alejandro Pozas-Kerstjens, Emanuel-Cristian Boghiu
"""
import copy
import itertools

import numpy as np
import sympy
# import warnings
from functools import lru_cache

from causalinflation.quantum.fast_npa import (factorize_monomial,
                                              mon_lessthan_mon,
                                              mon_lexsorted,
                                              to_canonical,
                                              to_name,
                                              mon_equal_mon,
                                              reverse_mon)
# from causalinflation.quantum.typing import ArrayMonomial, StringMonomial, IntMonomial


# from collections import defaultdict, deque
from itertools import permutations, product
# ncpol2sdpa >= 1.12.3 is required for quantum problems to work
# from ncpol2sdpa import flatten, generate_operators, generate_variables
# from ncpol2sdpa.nc_utils import apply_substitutions

#from typing import Dict, List, Tuple, Union, Any, Iterable #, NewType, TypeVar
from causalinflation.quantum.types import List, Dict, Tuple, Union, Any, Iterable

#Adding for hopefully-efficient moment matrix symmetrization
from sympy.combinatorics import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup



try:
    # import numba
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

# TODO build a proper typing system, maybe use classes?

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


def mul(lst: List) -> Any:
    """Multiply all elements of a list.

    Parameters
    ----------
    lst : List
        Input list with elements that have a supported '*' multiplication.

    Returns
    -------
    Any
        Product of all elements.

    Example
    -------
    >>> mul([2, 'A_1', 'B_2'])
    2*A_1*B_2
    """

    if type(lst[0]) == str:
        result = '*'.join(lst)
    else:
        result = 1
        for element in lst:
            result *= element
    return result



def apply_source_perm_monomial(monomial: np.ndarray,
                                source: int,
                                permutation: np.ndarray,
                                commuting: bool_,
                                lexorder
                                ) -> np.ndarray:
    """This applies a source swap to a monomial.

    We assume in the monomial that all operators COMMUTE with each other.

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
    permutation_plus = np.hstack(([0], permutation + 1))
    return apply_source_permplus_monomial(
        monomial,
        source,
        permutation_plus,
        commuting,
        lexorder
    )

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
    outs_per_input : List[int]
        The number of outcomes of each measurement for a given party
    name : str
        The name to be associated to the party

    Returns
    -------
    list
        The list of Sympy operators
    """

    new_factors = monomial.copy()
    new_factors[:, 1+source] = np.take(permutation_plus, new_factors[:, 1+source])
    # for i in range(monomial.shape[0]):
    #     if new_factors[i][1 + source] > 0:
    #         new_factors[i][1 + source] = permutation[
    #                                         new_factors[i][1 + source] - 1] + 1
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
    commuting : bool
        Whether the operators commute or not.


    Returns
    -------
    List[np.ndarray]
        List of operators with the specified source permuted.

    """
    permuted_op_list = []
    for monomial in columns:
        # if np.array_equal(monomial, np.array([[0]], dtype=np.uint16)):
        (row_count, col_count) = monomial.shape
        if row_count == 0 or col_count ==1 : #ACCOMODATING LEGACY IDENTITY OPERATOR
            permuted_op_list.append(monomial)
        else:
            newmon = apply_source_perm_monomial(monomial, source,
                                                np.asarray(permutation),
                                                commuting, lexorder)
            canonical = to_canonical(newmon, notcomm, lexorder)
            permuted_op_list.append(canonical)

    return permuted_op_list


@jit(nopython=nopython)
def apply_source_permutation_monomial(monomial: np.ndarray,
                                      source: int,
                                      permutation: np.ndarray
                                      ) -> np.ndarray:
    """Applies a source permutation to a single monomial.

    SPEED NOTE: if you want to apply a simple source swap as opposed
    to an arbitrary permutation, use apply_source_swap_monomial instead,
    as it is 25x faster.

    Parameters
    ----------
    monomial : np.ndarray
        Input monomial in 2d array format.
    source : int
        Source that is being swapped.
    permutation : np.ndarray
        Permutation of the copies of the specified source.

    Returns
    -------
    np.ndarray
        Monomial with the specified source permuted.
    """

    new_factors = monomial.copy()
    for i in range(len(new_factors)):
        if new_factors[i, 1 + source] > 0:
            # Python starts counting at 0
            new_factors[i, 1 + source] = permutation[
                                            new_factors[i, 1 + source] - 1] + 1
        else:
            continue

    return new_factors


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

    assert max_monomial_length <= min(inflevels),                              \
        ("You cannot have a longer list of commuting operators" +
         " than the inflation level.")

    # The initial monomial to which we will apply the symmetries
    # For instance, if max_monomial_length=4, this is something of the form
    # A_1_1_0_xa * A_2_2_0_xa * A_3_3_0_xa * A_4_4_0_xa
    initial_monomial = np.zeros(
        (max_monomial_length, 1+nr_sources+2), dtype=np.uint16)
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
                permuted, source, np.array(perms[source]), True, lexorder)
        permuted_name = to_name(permuted, names)
        if permuted_name not in template_new_mons_aux:
            template_new_mons_aux.append(permuted_name)

    template_new_monomials = [
        np.asarray(to_numbers(mon, names)) for mon in template_new_mons_aux]

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

    # this is for treating cases like A**2, where we want factors = [A, A]
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
        symbols.append(sympy.Symbol('_'.join([names[letter[0]-1]] +
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

    # the following commented code is compatible with numba, but it is slower
    # than native...
    # TODO see if possible to write a numba function that does strings
    # fast
    '''
    # The following is compatible with numba and can be precompiled however...
    # it's 1 microsec slower than the native python version!!
    # And it only work with single digit numbers, because int('2')
    # to get integer 2 is not supported by numba yet
    # https://github.com/numba/numba/issues/5723
    # That's very surprising!
    # native python version is is around 5 micro seconds for small inputs

    monomial_parts = monomial.split('*')
    monomial_parts_indices = np.zeros(shape=(len(monomial_parts),
                            len(monomial_parts[0].split('_'))),dtype=np.int8)
    monomial_parts_indices[1] = 2
    for i, part in enumerate(monomial_parts):
        atoms = part.split('_')

        monomial_parts_indices[i, 0] = ord(atoms[0]) - ord('A')

        for i2, j in enumerate(atoms[1:-2]):
            monomial_parts_indices[i, i2] = ord(j) - ord('0')
        monomial_parts_indices[i, -2] = ord(atoms[-2])-ord('0')
        monomial_parts_indices[i, -1] = ord(atoms[-1])-ord('0')
    '''

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

    return np.array(monomial_parts_indices, dtype=np.uint16) #Elie warning: Could the string '1' cause problems here??


def from_numbers_to_flat_tuples(lista: List[np.ndarray]
                                ) -> List[Tuple[int]]:
    """Flatten all monomials in the list represented as lists of lists to a
    flat tuple.

def factorize_monomials(monomials_as_numbers: np.ndarray,
                        verbose: int = 0
                        ) -> np.ndarray:
    """Applies factorize_momonial to each monomial in the input list
    of monomials.

    Parameters
    ----------
    monomials_as_numbers : numpy.ndarray
        An ndarray of type object where each row has the integer
        representation of a monomial in the 1st column and in the 2nd
        column a monomial in matrix form (each row is an operator
        and each column has the operator indices).
    verbose : int, optional
        Whether to print progress bar. By default ``0``.

    Returns
    -------
    numpy.ndarray
        The same output as monomials_as_numbers but with the factorized
        monomials.
    """
    monomials_factors = monomials_as_numbers.copy()
    for idx, [_, monomial] in enumerate(monomials_factors):
        monomials_factors[idx][1] = factorize_monomial(monomial)
    return monomials_factors


def is_knowable(monomial: np.ndarray) -> bool:
    """Determine whether a given atomic monomial (which cannot be factorized
    into smaller disconnected components) admits an identification with a
    monomial of the original scenario.

    Parameters
    ----------
    monomial_as_array : Union[List[List[int]], np.ndarray]
        List of operators, denoted each by a list of indices

    Returns
    -------
    bool
        Whether the monomial is knowable or not.
    """

    # After inflation and factorization, a monomial is known if it just
    # contains at most one operator per party, and in the case of having
    # one operator per node in the network, if the corresponding graph is
    # the same as the scenario hypergraph.

    # monomial_as_array = np.asarray(monomial)
    assert monomial_as_array.ndim == 2, "You must enter a list of monomials. Hence,"\
                        + " the number of dimensions of monomial must be 2"
    parties = monomial_as_array[:, 0].astype(int)
    # If there is more than one monomial of a party, it is not knowable
    if len(set(parties)) != len(parties):
        return False
    else:
       # We see if, for each source, there is at most one copy used
        return all([len(set(source[np.nonzero(source)])) <= 1
                    for source in monomial_as_array[:, 1:-2].T])


# @lru_cache(maxsize=None, typed=False)
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

    # monomial = np.array(monomial_in, dtype=np.int8).copy()
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

# @lru_cache(maxsize=None, typed=False)
# def atomic_is_physical_memoized(atomic_monomial_in: Tuple[Tuple[int]], **kwargs) -> bool:
#     return atomic_is_physical(np.asarray(atomic_monomial_in), **kwargs)
#


def label_knowable_and_unknowable(monomials_factors: np.ndarray,
                                  hypergraph: np.ndarray
                                  ) -> np.ndarray:
    """Given the list of monomials factorised, label each monomial as knowable,
    semiknowable or unknowable.

    Parameters
    ----------
    monomials_factors_input : numpy.ndarray
        Ndarray of factorised monomials. Each row encodes the integer
        representation and the factors of the monomial.
    hypergraph : numpy.ndarray
        The hypergraph of the network.

    Returns
    -------
    numpy.ndarray
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
        The list of all the symbolic factors, with the powers expanded.
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
    sym_variables_to_be_given = copy.deepcopy(variables_to_be_given)
    for idx, [_, term] in enumerate(variables_to_be_given):
        factors  = term.split('*')
        # nr_terms = len(factors)
        factors  = np.array([list(factor.split('_')) for factor in factors])
        parties  = factors[:, 0]
        inputs   = factors[:, -2]
        outputs  = factors[:, -1]
        name = 'p'
        # Add specification of parties if a marginal probability
        if len(parties) < max_nr_of_parties:
            name += ''.join(parties)
        name += '('
        name += ''.join(outputs)
        name += '|'
        name += ''.join(inputs)
        name += ')'
        sym_variables_to_be_given[idx][1] = sympy.symbols(name)

    return sym_variables_to_be_given


def substitute_sym_with_value(syminput: sympy.core.symbol.Symbol,
                              settings_per_party: Tuple[int],
                              outcomes_per_party: Tuple[int],
                              p_vector: np.ndarray
                              ) -> float:
    """Function which, given a symbolic probability in the form
    p(abc...|xyz...) and a probability distribution p called as
    p[a,b,c,...,x,y,z,...], returns the numerical value of the
    probability.

    Note that this accepts marginals, for example, p(a|x), and
    then it automatically computes all the summations over
    p[a,b,c,...,x,y,z,...].

    Parameters
    ----------
    syminput : sympy.core.symbol.Symbol
        Symbolic probability.
    settings_per_party : Tuple[int]
        Setting cardinalities per party.
    outcomes_per_party : Tuple[int]
        Outcome cardinalities per party.
    p_vector : np.ndarray
        The probability distribution of dims
        (outcomes_per_party,settings_per_party).

    Returns
    -------
    float
        The value of the symbolic probability (which can be a marginal)

    Examples
    --------
    >>> p = sympy.symbols('pA(0|1)')
    >>> substitute_sym_with_value(p, (2,2), (2,2), p_vector)
    parray[0,:,1,0].sum()

    Note that we take the first setting (=0) for marginalised parties, in the
    example above, the second party is marginalised.
    """

    # Extract the parties
    nrparties = len(settings_per_party)
    name = syminput.name
    charelement = name[0]  # should be 'p'
    assert charelement == 'p', ("The names of the symbolic variables" +
                                                        " are not correct.")
    parties = []  # Parties over which to NOT marginalize.
    idx = 1
    if name[1] == '(':
        parties = [chr(ord('A') + i) for i in range(nrparties)]
    else:
        while name[idx] != '(':
            parties.append(name[idx])
            idx += 1
    assert parties == sorted(parties), ("The symbolic variables should " +
                                    "have the parties in the correct order.")
    idx += 1
    outcomes = []
    while name[idx] != '|':
        outcomes.append(int(name[idx]))
        idx += 1
    idx += 1
    inputs = []
    while name[idx] != ')':
        inputs.append(int(name[idx]))
        idx += 1

    # Assume parties are in order 'A'->0, 'B'->1, 'C'->2, etc.
    parties_idx = [ord(p) - ord('A') for p in parties]
    if parties_idx:  # if not empty
        aux = list(range(nrparties))
        for p in parties_idx:
            aux.remove(p)
        over_which_to_marginalize = aux
    else:
        over_which_to_marginalize = []

    # Because of no signaling, when marginalising over a party,
    # its input does not affect the marginal. However, when
    # going from the marginal to the full distribution, we
    # must choose an input for the marginalised party. By default
    # we choose input 0.
    # eg pA(a|x)->pA(a|x,y=0,z=0)
    settings_aux = [[0] for _ in range(nrparties)]
    i_idx = 0
    for p in parties_idx:
        settings_aux[p] = [inputs[i_idx]]
        i_idx += 1
    # For the outcomes, we define a list of lists where the outcomes
    # that are not marginalized over have a fixed value; for the
    # others we give all possible values.
    # example: pAC(0,1|1,2) --> [[0],[0,1,2],[1]]
    # we have [0,1,2] because Bob is being marginalized over, so we put
    # all outcome values, but we only put 0 and 1 for Alice and Charlie.
    # This construction allows to easily use itertools.product to do
    # the marginalisation on parray over the marginalised parties.
    outcomes_aux = []
    for p in range(nrparties):
        if p in parties_idx:
            # i use .index in case the parties are disordered
            outcomes_aux.append([outcomes[parties_idx.index(p)]])
        else:
            tuples.append(tuple(flatten(element.tolist())))
    return tuples


def monomialset_name2num(monomials: np.ndarray, names: List[str]) -> np.ndarray:
    """Change each monomial in the list of monomials from string to matrix
    representation.

    Parameters
    ----------
    monomials : numpy.ndarray
        Input monomials list.
    names : List[str]
         List of party names

    Returns
    -------
    numpy.ndarray
        The list of input monomials, but with each monomial in matrix
        representation.
    """
    monomials_numbers = monomials.copy()
    for i, line in enumerate(monomials):
        monomials_numbers[i][1] = to_numbers(line[1], names)
    monomials_numbers[:, 0] = monomials_numbers[:, 0].astype(int)
    return monomials_numbers


def monomialset_num2name(monomials_factors: np.ndarray,
                         names: List[str]
                         ) -> np.ndarray:
    """Change the list of monomials from a list of arrays to a list of strings.

    Parameters
    ----------
    monomials_factors : numpy.ndarray
        List of monomials.
    names : List[str]
        names[i] is the name of party i+1 (parties in [1,2,3,4...]).

    Returns
    -------
    numpy.ndarray
        The input with the monomials replaced by their string representation.
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
    #if len(parties) < max_nr_of_parties:
    name += '_{'
    for p in parties:
        name += p
    name += '}'
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
        The monomial in list of lists format (equivalent to 2d array format by
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


# @jit(nopython=nopython)  # make to_canonical compatible with numba
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
        np.asarray(monomial_component), notcomm, lexorder)  # Make sure all commutation rules are applied
    new_mon = monomial_component.copy()
    # -2 we ignore the first and the last two columns
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

# @jit(nopython=nopython)  # remove the use of itertools
def to_repr_swap_plus_commutation(mon_aux: np.ndarray,
                                  inflevels: np.ndarray,
                                  notcomm: np.ndarray,
                                  lexorder: np.ndarray,
                                  commuting: bool) -> np.ndarray:
    # Now we must take into account that the application of symmetries plus
    # applying commutation rules can give us an even smaller monomial
    # (lexicographically). To deal with this, we will apply *all* possible
    # source swaps and then apply commutation rules, and if the resulting
    # monomial is smaller, we accept it.
    # TODO: This is horribly inefficient if we have lots of symmetries
    # TODO 2: to make it compatible with numba, avoid using itertools!
    # (i.e., product, permutations)
    nr_sources = inflevels.shape[0]
    all_perms_per_source = [permutations(range(inflevels[source]))
                            for source in range(nr_sources)]

    final_monomial = mon_aux.copy()
    prev = mon_aux
    while True:
        for perms in product(*all_perms_per_source):
            permuted = final_monomial.copy()
            for source in range(nr_sources):
                permuted = apply_source_perm_monomial(permuted,
                                                      source,
                                                      np.asarray(perms[source]),
                                                      commuting,
                                                      lexorder)
            permuted = to_canonical(permuted, notcomm, lexorder)  #ALREADY DONE by apply_source_perm_monomial ??
            if mon_lessthan_mon(permuted, final_monomial, lexorder):
                final_monomial = permuted
        if np.array_equal(final_monomial, prev):
            break
        prev = final_monomial

    return final_monomial


# @jit(nopython=nopython)
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
    # lexiographic ordering. We do this by lowering the copy indices making
    # them as low as possible from left to right. This might not be a global
    # optimum.
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


def substitute_sym_with_numbers(symbolic_variables_to_be_given: List[List[sympy.core.symbol.Symbol]],
                                settings_per_party: Tuple[int],
                                outcomes_per_party: Tuple[int],
                                p_vector: np.ndarray
                                ) -> List[List[float]]:
    """Substitute all symbolic variables of the form 'p(ab..|xy..)' with
    the corresponding value or marginal computed from p_vector.

    input_list       = np.zeros(n_parties, dtype=int)
    input_list[dont_marginalize] = inputs
    inputs_outputs   = outputs + input_list.tolist()
    return marginal_dist[tuple(inputs_outputs)]


    Returns
    -------
    List[Tuple[int, float]]
        A nested list of type [..., [int, float], ...] where every
        symbolic probability in the input is substituted with its numerical
        value.
    """
    variables_values = copy.deepcopy(symbolic_variables_to_be_given)
    for i in range(len(variables_values)):
        variables_values[i][1] = float(substitute_sym_with_value(
                                           symbolic_variables_to_be_given[i][1],
                                                             settings_per_party,
                                                             outcomes_per_party,
                                                             p_vector))
    return variables_values


def clean_coefficients(coefficients: np.ndarray,
                       chop_tol: float=1e-10,
                       round_decimals: int=3) -> np.ndarray:
    """Clean the list of coefficients in a certificate.

    Parameters
    ----------
    coefficients : np.ndarray
      The list of coefficients.
    chop_tol : float, optional
      Coefficients in the dual certificate smaller in absolute value are
      set to zero. Defaults to 1e-10.
    round_decimals : int, optional
      Coefficients that are not set to zero are rounded to the number
      of decimals specified. Defaults to 3.

    Returns
    -------
    np.ndarray
      The cleaned-up coefficients.
    """
    coeffs = copy.deepcopy(coefficients)
    # Set to zero very small coefficients
    coeffs[np.abs(coeffs) <= chop_tol] = 0
    # Take the biggest one and make it 1
    normalising_factor = np.max(np.abs(coeffs[np.abs(coeffs) > chop_tol]))
    coeffs /= normalising_factor
    # Round
    coeffs = np.round(coeffs, decimals=round_decimals)
    return coeffs


def dimino_sympy(group_generators):
    gens = [Permutation(list(gen)) for gen in group_generators]
    group = PermutationGroup(gens)
    group_elements = list(group.generate_dimino(af=True))
    return group_elements


def compute_numeric_value(mon_string: str,
                          p_array: np.ndarray,
                          parties_names: List[str]
                          ) -> float:
    """Function which, given a monomial and a probability distribution p_array
    called as p_array[a,b,c,...,x,y,z,...], returns the numerical value of the
    probability associated to the monomial.

    Note that this accepts marginals, for example, p(a|x), and then it
    automatically computes all the summations over p[a,b,c,...,x,y,z,...].

    Parameters
    ----------
    mon_string : String
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
    >>> compute_numeric_value(p, [2,2], [2,2], p_array)
    parray[o,:,i,0].sum()
    Note that we take the first setting (=0) for marginalised parties, in the
    example above, the second party is marginalised.
    """
    if mon_string == '1':
        return 1.0
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


def flatten(nested):
    """Keeps flattening a nested lists of lists until  the
    first element of the resulting list is not a list.
    """
    if isinstance(nested, np.ndarray):
        return nested.ravel().tolist()
    else:
        while isinstance(nested[0], Iterable):
            nested = list(itertools.chain.from_iterable(nested))
            # nested = [item for sublist in nested for item in sublist]
        return nested


# def generate_noncommuting_measurements(nro_per_input, name):
#     """Rewritten function from ncpol2sdpa but without using the
#     quantum operators from Sympy, but just general NC variables.
#     """
#     ops_per_input = []
#     for x, nro in enumerate(nro_per_input):
#         ops_per_output_per_input = []
#         for o in range(nro):
#             ops_per_output_per_input.append(
#                 sympy.Symbol(name + '_' + str(x) + '_' + str(o), commutative=False)
#             )
#         ops_per_input.append(ops_per_output_per_input)
#     return ops_per_input

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
