# from __future__ import annotations
# from __future__ import absolute_import
# from __future__ import with_statement
import numpy as np
from causalinflation.quantum.general_tools import factorize_monomial, is_physical, is_knowable  # to_representative_aux
# from causalinflation.quantum.fast_npa import mon_equal_mon
# import itertools
# from functools import cached_property
from functools import lru_cache
from functools import total_ordering

from typing import List, Tuple, Union
from collections.abc import Iterable
# ListOrTuple = NewType("ListOrTuple", Union[List, Tuple])
# MonomInputType = NewType("NumpyCompat", Union[np.ndarray, ListOrTuple[ListOrTuple[int]]])

import sympy
# from itertools import chain
from operator import attrgetter



def to_tuple_of_tuples(monomial: np.ndarray) -> tuple:
    if isinstance(monomial, tuple):
        return monomial
    elif isinstance(monomial, np.ndarray):
        if monomial.ndim >= 3:
            return tuple(to_tuple_of_tuples(operator) for operator in monomial) 
        elif monomial.ndim == 2:
            # NOTE: This is to speed up the function for the frequenct case of 
            # 2d arrays. Adding this special clause speeds up by 5x the 2d case. 
            # Note: Using "map" on my CPU takes is slightly faster than list
            # comprehension, i.e., tuple(tuple(l) for l in monomial.tolist()).
            return tuple(map(tuple, monomial.tolist())) 
        elif monomial.ndim == 1:
            return tuple(monomial.tolist())
        else:
            return monomial.tolist()
    elif isinstance(monomial, Iterable):
        return to_tuple_of_tuples(np.array(monomial))
    else:
        return monomial


@lru_cache(maxsize=None, typed=False)
def compute_marginal_memoized(prob_array: Tuple, atom: Tuple[Tuple[int]]) -> float:
    return compute_marginal(np.asarray(prob_array), np.asarray(atom))


def compute_marginal(prob_array: np.ndarray, atom: np.ndarray) -> float:
    """Function which, given an atomic monomial and a probability distribution prob_array
        called as prob_array[a,b,c,...,x,y,z,...], returns the numerical value of the
        probability associated to the monomial.
        The atomic monomial is a list of length-3 vectors.
        The first element indicates the party,
        the second element indicates the setting,
        the final element indicates the outcome.
        Note that this accepts marginals and then
        automatically computes all the summations over p[a,b,c,...,x,y,z,...].
        Parameters
        ----------

        prob_array : np.ndarray
            The probability distribution of dims
            (outcomes_per_party, settings_per_party).
        atom : np.ndarray
            Monomial indicated a (commuting) collection of measurement operators.
        Returns
        -------
        float
            The value of the symbolic probability (which can be a marginal)
        """
    if len(atom):
        n_parties: int = prob_array.ndim // 2
        participating_parties = atom[:, 0] - 1  # Convention in numpy monomial format is first party = 1
        inputs = atom[:, -2].astype(int)
        outputs = atom[:, -1].astype(int)
        indices_to_sum = list(set(range(n_parties)).difference(participating_parties))
        marginal_dist = np.sum(prob_array, axis=tuple(indices_to_sum))
        input_list: np.ndarray = np.zeros(n_parties, dtype=int)
        input_list[participating_parties] = inputs
        outputs_inputs = np.concatenate((outputs, input_list))
        return marginal_dist[tuple(outputs_inputs)]
    else:
        return 1.


def atomic_monomial_to_name(observable_names: Tuple[str],
                            atom: Union[np.ndarray, Tuple[Tuple[int]]],
                            human_readable_over_machine_readable=True) -> str:
    atom_as_array = np.array(atom, dtype=int, copy=True)
    if len(atom_as_array):
        if human_readable_over_machine_readable:
            if atom_as_array.shape[-1] == 3: #this handles the KNOWN factors.
                parties = np.take(observable_names,
                                  atom_as_array[:, 0] - 1)  # Convention in numpy monomial format is first party = 1
                inputs = [str(input) for input in atom_as_array[:, -2].astype(int).tolist()]
                outputs = [str(output) for output in atom_as_array[:, -1].astype(int).tolist()]
                p_divider = '' if all(len(p) == 1 for p in parties) else ','
                # We will probably never have more than 1 digit cardinalities
                # but who knows...
                i_divider = '' if all(len(i) == 1 for i in inputs) else ','
                o_divider = '' if all(len(o) == 1 for o in outputs) else ','
                return ('p_{' + p_divider.join(parties) + '}' +
                        '(' + o_divider.join(outputs) + '|' + i_divider.join(inputs) + ')')
            else:
                operators_as_strings = []
                for op in atom_as_array.tolist(): #this handles the UNKNOWN factors.
                    operators_as_strings.append('_'.join([observable_names[op[0] - 1]]  # party idx
                                                         + [str(i) for i in op[1:]]))
                return 'P[' + ', '.join(operators_as_strings) + ']'
        else:
            res = '*'.join([observable_names[op[0] - 1] + '_' +
                             '_'.join([str(i) for i in op[1:]])
                             for op in atom])
            return res
    else:
        return '1'

def symbol_from_atomic_name(atomic_name):
    if atomic_name == '1':
        return sympy.S.One
    elif atomic_name == '0':
        return sympy.S.Zero
    else:
        return sympy.Symbol(atomic_name, commutative=True)


def name_from_atomic_names(atomic_names):
    return '*'.join(atomic_names)


def symbol_from_atomic_names(atomic_names):
    prod = sympy.S.One
    for op_name in atomic_names:
        if op_name != '1':
            prod *= sympy.Symbol(op_name,
                                 commutative=True)
    return prod

def symbol_from_atomic_symbols(atomic_symbols):
    prod = sympy.S.One
    for sym in atomic_symbols:
        prod *= sym
    return prod

# Elie comment: using class memoization via https://stackoverflow.com/questions/47785795/memoized-objects-still-have-their-init-invoked
class AtomicMonomialMeta(type):
    def __init__(self, name, bases, namespace):
        super().__init__(name, bases, namespace)
        self.cache = dict()
    def __call__(self, array2d: Union[np.ndarray, Tuple[Tuple[int]], List[List[int]]], **kwargs):
        quick_key = to_tuple_of_tuples(array2d)
        if quick_key not in self.cache:
            # print('New Instance')
            self.cache[quick_key] = super().__call__(array2d, **kwargs)
        else:
            # print('Existing Instance')
            return self.cache[quick_key]

@total_ordering
class AtomicMonomial(metaclass=AtomicMonomialMeta):
    __slots__ = ['as_ndarray',
                 'n_ops',
                 'op_length',
                 'as_tuples',
                 'signature',
                 'inflation_indices_are_irrelevant',
                 'knowable_q',
                 'knowability_status',
                 'known_status',
                 'known_value',
                 'name',
                 'symbol']
    def __init__(self, array2d: Union[np.ndarray, Tuple[Tuple[int]], List[List[int]]],
                 atomic_is_knowable=is_knowable):
        """
        This class is incredibly inefficient unless knowable_q has built-in memoization.
        It is designed to categorize monomials into known, semiknown, unknown, etc.
        Note that if knowable_q changes (such as given partial information) we can update this on the fly.
        """
        # self.to_representative = lambda mon: tuple(tuple(vec) for vec in to_representative(mon))

        self.as_ndarray = np.asarray(array2d)
        if len(self.as_ndarray):
            assert self.as_ndarray.ndim == 2, 'Expected 2 dimension numpy array.'
        else:
            self.as_ndarray = np.empty((0, 3), dtype=np.uint8)
        self.n_ops, self.op_length = self.as_ndarray.shape
        assert self.op_length >= 3, 'Expected at least 3 digits to specify party, outcome, settings.'
        self.inflation_indices_are_irrelevant = is_knowable(self.as_ndarray)
        self.knowable_q = self.inflation_indices_are_irrelevant and atomic_is_knowable(self.as_ndarray)
        if self.knowable_q:
            self.knowability_status = 'Yes'
        else:
            self.knowability_status = 'No'
        self.known_value = 1.
        self.known_status = False
        if self.inflation_indices_are_irrelevant:
            self.as_tuples = to_tuple_of_tuples(np.take(self.as_ndarray, [0, -2, -1], axis=1))
        else:
            self.as_tuples = to_tuple_of_tuples(self.as_ndarray)
        self.signature = self.as_tuples

    def __str__(self):
        # If a human readable name is available, we use it.
        try:
            return self.name
        except AttributeError:
            return np.array2string(self.as_ndarray)

    def __repr__(self):
        return self.__str__()

    def update_hash_via_to_representative_function(self, to_representative_function):
        if not self.inflation_indices_are_irrelevant:
            self.as_ndarray = to_representative_function(self.as_ndarray)
            self.as_tuples = to_tuple_of_tuples(self.as_ndarray)
            self.signature = self.as_tuples

    def __hash__(self):
        # return hash(tuple(sorted([to_tuple_of_tuples(self.to_representative_function(factor)) for factor in self.factors])))
        return hash(self.signature)

    def __eq__(self, other):
        return self.signature == other.signature
    def __lt__(self, other):
        return self.signature < other.signature


    def update_name_and_symbol_given_observed_names(self, observable_names):
        self.name = atomic_monomial_to_name(observable_names=observable_names,
                                                                 atom=self.as_ndarray,
                                                                 human_readable_over_machine_readable=True)
        self.symbol = symbol_from_atomic_names([self.name])





class Monomial(object):
    __slots__ = ['as_ndarray',
                 'n_ops',
                 'op_length',
                 'as_tuples',
                 'is_physical',
                 'factors',
                 'factors_as_atomic_monomials',
                 'signature',
                 'is_atomic',
                 'nof_factors',
                 'atomic_knowability_status',
                 'knowable_factors_uncompressed',
                 'knowable_factors',
                 'unknowable_factors',
                 'unknowable_factors_as_block',
                 '_unknowable_block_len',
                 'knowability_status',
                 'knowable_q',
                 'physical_q',
                 'idx',
                 'mask_matrix',
                 'known_status',
                 'known_value',
                 'unknown_part',
                 'representative',
                 'unknown_signature',
                 'irrelevant_copy_indices_status',
                 'factors_with_irrelevant_copy_indices',
                 'factors_with_relevant_copy_indices',
                 '_factors_repr', # Emi to Elie: See comment the 'set_values' method
                 # 'knowable_factors_names',
                 # 'knowable_part_name',
                 # 'knowable_part_machine_readable_name',
                 # 'unknowable_part_name',
                 # 'unknowable_part_machine_readable_name',
                 '_names_of_factors',
                 # '_machine_readable_names_of_factors',
                 'name',
                 'machine_readable_name',
                 'symbol',
                 'machine_readable_symbol',
                 ]

    def __init__(self, array2d: Union[np.ndarray, Tuple[Tuple[int]], List[List[int]]],
                 atomic_is_knowable=is_knowable,
                 sandwich_positivity=False,
                 idx=0,
                 to_representative_function=np.asarray):
        """
        This class is incredibly inefficient unless knowable_q has built-in memoization.
        It is designed to categorize monomials into known, semiknown, unknown, etc.
        Note that if knowable_q changes (such as given partial information) we can update this on the fly.
        """
        # self.to_representative = lambda mon: tuple(tuple(vec) for vec in to_representative(mon))

        self.idx = idx
        self.as_ndarray = np.asarray(array2d)
        if len(self.as_ndarray):
            assert self.as_ndarray.ndim == 2, 'Expected 2 dimension numpy array:' + np.array2string(self.as_ndarray)
        else:
            self.as_ndarray = np.empty((0, 3), dtype=np.uint8)
        self.n_ops, self.op_length = self.as_ndarray.shape
        assert self.op_length >= 3, 'Expected at least 3 digits to specify party, outcome, settings.'

        self.factors = factorize_monomial(self.as_ndarray, canonical_order=False)
        self.nof_factors = len(self.factors)
        self.is_atomic = (self.nof_factors <= 1)
        self.factors_as_atomic_monomials = [AtomicMonomial(factor, atomic_is_knowable=atomic_is_knowable) for factor in self.factors]
        if self.nof_factors == 1:
            self.signature = self.factors_as_atomic_monomials[0] #So as to match atomic signature
        else:
            self.signature = tuple(sorted(self.factors_as_atomic_monomials))
        self.unknown_signature = self.signature



        #TODO: Elie to Alex: I'm in the middle of a large overhaul to make use of AtomicMonomial objects.

        self.atomic_knowability_status = tuple(atomic_is_knowable(atom) for atom in self.factors)

        self.knowable_factors_uncompressed = tuple(atom for atom, knowable in
                                                   zip(self.factors,
                                                       self.atomic_knowability_status)
                                                   if knowable)
        self.unknowable_factors = tuple(atom for atom, knowable in
                                        zip(self.factors,
                                            self.atomic_knowability_status)
                                        if not knowable)
        self.unknowable_factors_as_block = self.factors_as_block(self.unknowable_factors)
        # self.knowable_factors = tuple(tuple(tuple(vec) for vec in np.take(factor, [0, -2, -1], axis=1))
        #         for factor in self.knowable_factors_uncompressed)
        # knowable factors must be hashable, but this is taken care of after externally rectifying fake settings.
        self.knowable_factors = tuple(np.take(factor, [0, -2, -1], axis=1).astype(int)
                                      for factor in self.knowable_factors_uncompressed)

        #We distinguish between factor for which the copy indices are relevant, and factors for which they are not.
        self.irrelevant_copy_indices_status = tuple(is_knowable(atom) for atom in self.unknowable_factors)
        self.factors_with_relevant_copy_indices = tuple(factor for factor, copy_index_irrelevance in
                                                        zip(self.unknowable_factors,
                                                            self.irrelevant_copy_indices_status)
                                                        if not copy_index_irrelevance)
        self.factors_with_irrelevant_copy_indices = tuple(np.take(factor, [0, -2, -1], axis=1)
                    for factor, copy_index_irrelevance in
                    zip(self.unknowable_factors,
                        self.irrelevant_copy_indices_status)
                    if copy_index_irrelevance)
        self.factors_with_irrelevant_copy_indices = self.knowable_factors + self.factors_with_irrelevant_copy_indices
        self.factors_with_irrelevant_copy_indices = tuple(sorted(to_tuple_of_tuples(factor) for factor in self.factors_with_irrelevant_copy_indices))

        # self.as_tuples = self.factors_with_irrelevant_copy_indices + tuple(
        #     sorted([to_tuple_of_tuples(factor) for factor in self.factors_with_relevant_copy_indices]))
        self.as_tuples = to_tuple_of_tuples(self.as_ndarray)

        self.knowable_q = (len(self.knowable_factors) == self.nof_factors)
        self.physical_q = self.knowable_q or is_physical(self.unknowable_factors_as_block,
                                                         sandwich_positivity=sandwich_positivity)
        self._unknowable_block_len = len(self.unknowable_factors_as_block)
        # self.to_representative_function = to_representative_function

        if self._unknowable_block_len == 0:
            self.knowability_status = 'Yes'
        elif self._unknowable_block_len == self.n_ops:
            self.knowability_status = 'No'
        else:
            self.knowability_status = 'Semi'

        if self.idx > 1:
            self.known_status = 'No'
        else:
            self.known_status = 'Yes'
        self.unknown_part = self.as_ndarray
        self.known_value = 1.

    def __str__(self):
        #If a human readable name is available, we use it.
        try:
            return self.name
        except AttributeError:
            return np.array2string(self.as_ndarray)

    def __repr__(self):
        return self.__str__()

    def update_atomic_constituents(self, to_representative_function):
        for atomic_mon in self.factors_as_atomic_monomials:
            atomic_mon.update_hash_via_to_representative_function(to_representative_function) #Automatically changes self.factors_as_atomic_monomials
        if self.nof_factors == 1:
            self.signature = self.factors_as_atomic_monomials[0]
        else:
            self.signature = tuple(sorted(self.factors_as_atomic_monomials))

    def update_hash_via_to_representative_function(self, to_representative_function):
        # self.factors_with_relevant_copy_indices = tuple(to_representative_function(factor) for factor in self.factors_with_relevant_copy_indices)
        # # WARNING: WE CANNOT STACK THE FACTORS AFTER CONVERTING TO REPRESENTATIVE!
        # # The alternative is to call to_representative_function on the entire as_ndarray, instead of on the individual factors.
        # self.as_tuples = self.factors_with_irrelevant_copy_indices + tuple(sorted(
        #     to_tuple_of_tuples(factor) for factor in self.factors_with_relevant_copy_indices))
        self.as_ndarray = to_representative_function(self.as_ndarray)
        self.as_tuples = to_tuple_of_tuples(self.as_ndarray)


    def __hash__(self):
        # return hash(tuple(sorted([to_tuple_of_tuples(self.to_representative_function(factor)) for factor in self.factors])))
        return hash(self.signature)

    def __eq__(self, other):
        return self.signature == other.signature
    def __lt__(self, other):
        return self.signature < other.signature

    # def __eq__(self, other):
    #     if isinstance(other, Monomial):
    #         # TODO: What if they have different nd_arrays BUT the same
    #         # representative factors? This is currently not being done
    #         # correctly..
    #         return mon_equal_mon(self.as_ndarray, other.as_ndarray)
    #     elif isinstance(other, tuple):
    #         # TODO: Should we actually allow for this comparison??
    #         # If we do NOT allow for this comparison, then we need to
    #         # be able to instantiante Monomials very easily from tuples.
    #         # However right now when instantiating Monomials, certain things
    #         # are calculated like the factors or knowability status, making
    #         # initialisation slow. For now, I will just allow for comparison
    #         # between tuples and monomials.
    #         return self.as_tuples == other
    #     elif isinstance(other, list):
    #         return self.as_tuples == tuple(map(tuple, other))
    #     elif isinstance(other, np.ndarray):
    #         return mon_equal_mon(self.as_ndarray, other)
    #     else:
    #         raise Exception("Don't know how to compare {} to {}".format(self, other))


    def factors_as_block(self, factors):
        if len(factors):
            return np.concatenate(factors)
        else:
            return np.empty((0, self.op_length), dtype=np.int8)

    # knowability_status should not be used. Only after set_distribution is this relevant!
    # @cached_property
    # def knowability_status(self):
    #     if len(self.knowable_factors) == self.nof_factors:
    #         return 'Yes'
    #     elif len(self.knowable_factors) > 0:
    #         return 'Semi'
    #     else:
    #         return 'No'

    def update_given_valuation_of_knowable_part(self, valuation_of_knowable_part):
        actually_known_factors = np.logical_not(np.isnan(valuation_of_knowable_part))
        self.known_value = float(np.prod(np.compress(
            actually_known_factors,
            valuation_of_knowable_part)))
        # nof_known_factors = np.count_nonzero(actually_known_factors)
        knowable_factors_which_are_not_known = [factor for factor, known in
                                                zip(self.knowable_factors_uncompressed,
                                                    actually_known_factors)
                                                if not known]

        self.unknown_part = np.concatenate((
            self.factors_as_block(knowable_factors_which_are_not_known),
            self.unknowable_factors_as_block))
        unknown_len = len(self.unknown_part)
        if unknown_len == 0:
            self.known_status = 'Yes'
        elif unknown_len == self.n_ops:
            self.known_status = 'No'
        else:
            self.known_status = 'Semi'

    #TODO: Function is WORK IN PROGRESS!
    def update_given_some_atomic_monomials(self, set_of_known_atomic_monomials, remove_bad_values=False):
        if remove_bad_values:
            set_of_known_atomic_monomials = set(atomic_mon for atomic_mon in set_of_known_atomic_monomials if (atomic_mon.known_status and (not np.isnan(atomic_mon.known_value))))
        relevant_atomic_known_monomials = set(atomic_mon for atomic_mon in set_of_known_atomic_monomials if atomic_mon in self.factors_as_atomic_monomials)
        known_value = 1
        for mon in relevant_atomic_known_monomials:
            known_value *= mon.known_value
        raw_unknown_signature = tuple(sorted(mon for mon in self.factors_as_atomic_monomials if mon not in set_of_known_atomic_monomials))
        unknown_len = len(self.unknown_signature)
        if unknown_len == 0:
            self.known_status = 'Yes'
        elif unknown_len == self.n_ops:
            self.known_status = 'No'
        else:
            self.known_status = 'Semi'
        if unknown_len == 1:
            self.unknown_signature = raw_unknown_signature[0] #treat atomic cases special
        else:
            self.unknown_signature = raw_unknown_signature

    def update_name_and_symbol_given_observed_names(self, observable_names):
        self._names_of_factors = sorted([atomic_monomial_to_name(observable_names=observable_names,
                                                          atom=atomic_factor,
                                                          human_readable_over_machine_readable=True)
                                  for atomic_factor in self.knowable_factors], key=len) + sorted([
                                     atomic_monomial_to_name(observable_names=observable_names,
                                                             atom=atomic_factor,
                                                             human_readable_over_machine_readable=True)
                                     for atomic_factor in self.unknowable_factors], key=len)
        self.name = name_from_atomic_names(self._names_of_factors)
        self.symbol = symbol_from_atomic_names(self._names_of_factors)

        
        self.machine_readable_name = atomic_monomial_to_name(observable_names=observable_names,
                                            atom=self.as_ndarray,
                                            human_readable_over_machine_readable=False)

        self.machine_readable_symbol = (sympy.S.One if self.machine_readable_name == '1' else
                                        np.prod([sympy.Symbol(op, commutative=False)
                                                 for op in self.machine_readable_name.split('*')]) )
        

    def update_given_prob_dist(self, prob_array):
        if prob_array is None:
            self.known_status = 'No'
            self.known_value = 1
            self.unknown_part = self.as_ndarray
        else:
            hashable_prob_array = to_tuple_of_tuples(prob_array)
            valuation_of_knowable_part = np.array([
                compute_marginal_memoized(hashable_prob_array, atom)
                for atom in self.knowable_factors])
            return self.update_given_valuation_of_knowable_part(valuation_of_knowable_part)
    

    def to_symbol(self, objective_compatible=False):
        if objective_compatible:
            return self.machine_readable_symbol
        else:
            return self.symbol
