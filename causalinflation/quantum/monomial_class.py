# from __future__ import annotations
# from __future__ import absolute_import
# from __future__ import with_statement
import warnings

import numpy as np
from causalinflation.quantum.general_tools import factorize_monomial, is_physical, is_knowable  # to_representative_aux
# from causalinflation.quantum.fast_npa import mon_equal_mon
# import itertools
# from functools import cached_property
from functools import lru_cache, total_ordering  # , cached_property

from typing import List, Tuple, Union, Dict
from collections.abc import Iterable
from collections import Counter
# ListOrTuple = NewType("ListOrTuple", Union[List, Tuple])
# MonomInputType = NewType("NumpyCompat", Union[np.ndarray, ListOrTuple[ListOrTuple[int]]])

import sympy


# from itertools import chain
# from operator import attrgetter


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
                            human_readable_over_machine_readable=True,
                            do_conditional=False) -> str:
    atom_as_array = np.array(atom, dtype=int, copy=True)
    if len(atom_as_array):
        if human_readable_over_machine_readable:
            if atom_as_array.shape[-1] == 3:  # this handles the KNOWN factors.
                parties = np.take(observable_names,
                                  atom_as_array[:, 0] - 1)  # Convention in numpy monomial format is first party = 1
                inputs = [str(input) for input in atom_as_array[:, -2].astype(int).tolist()]
                outputs = [str(output) for output in atom_as_array[:, -1].astype(int).tolist()]
                p_divider = '' if all(len(p) == 1 for p in parties) else ','
                # We will probably never have more than 1 digit cardinalities
                # but who knows...
                i_divider = '' if all(len(i) == 1 for i in inputs) else ','
                o_divider = '' if all(len(o) == 1 for o in outputs) else ','
                if do_conditional:
                    return ('p_{' + p_divider.join(parties) + '}' +
                            '(' + o_divider.join(outputs) + ' do: ' + i_divider.join(inputs) + ')')
                else:
                    return ('p_{' + p_divider.join(parties) + '}' +
                            '(' + o_divider.join(outputs) + '|' + i_divider.join(inputs) + ')')
            else:
                operators_as_strings = []
                for op in atom_as_array.tolist():  # this handles the UNKNOWN factors.
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
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls.cache = dict()

    def __call__(cls, array2d: Union[np.ndarray, Tuple[Tuple[int]], List[List[int]]], **kwargs):
        quick_key = to_tuple_of_tuples(array2d)
        if quick_key not in cls.cache:
            # print('New Instance')
            cls.cache[quick_key] = super().__call__(array2d, **kwargs)
        return cls.cache[quick_key]


# TODO: We'd like to remove the distribution-dependant properties from the monomial itself, and instead use external dictionaries.


@total_ordering
class AtomicMonomial(metaclass=AtomicMonomialMeta):
    __slots__ = ['as_ndarray',
                 'rectified_ndarray',
                 # 'rectified_ndarray_as_tuples',
                 'not_yet_updated_by_to_representative',
                 'not_yet_fake_setting_rectified',
                 'n_ops',
                 'op_length',
                 # 'as_tuples',
                 # 'signature',
                 'inflation_indices_are_irrelevant',
                 'knowable_q',
                 'do_conditional',
                 'physical_q',
                 # 'knowability_status',
                 # 'known_status',
                 # 'known_value',
                 'not_yet_named',
                 'name',
                 'symbol',
                 'machine_name',
                 'machine_symbol']

    def __init__(self, array2d: Union[np.ndarray, Tuple[Tuple[int]], List[List[int]]],
                 atomic_is_knowable=is_knowable,
                 sandwich_positivity=False):
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
        # TODO: Note that irrelevance of inflation indices could be a different from knowable, since we can allow for noncommutation due to settings...
        self.inflation_indices_are_irrelevant = is_knowable(self.as_ndarray)
        self.knowable_q = self.inflation_indices_are_irrelevant and atomic_is_knowable(self.as_ndarray)
        self.do_conditional = self.inflation_indices_are_irrelevant and (not self.knowable_q)
        # if self.knowable_q:
        #     self.knowability_status = 'Yes'
        # else:
        #     self.knowability_status = 'No'
        self.physical_q = is_physical(self.as_ndarray, sandwich_positivity=sandwich_positivity)
        # self.known_value = 1.
        # self.known_status = False
        if self.inflation_indices_are_irrelevant:
            self.rectified_ndarray = np.take(self.as_ndarray, [0, -2, -1], axis=1)
        else:
            self.rectified_ndarray = self.as_ndarray
        # self.as_tuples = to_tuple_of_tuples(self.rectified_ndarray)  # OK, but later defined in terms of as_ndarray
        # self.rectified_ndarray_as_tuples = to_tuple_of_tuples(self.rectified_ndarray)
        self.not_yet_fake_setting_rectified = True
        self.not_yet_updated_by_to_representative = True
        self.not_yet_named = True

    @property
    def signature(self):
        """
        Signature of AtomicMonomial must match signature of CompoundMonomial when atomic.
        CompoundMonomial uses tuple(sorted(list_of_atoms)), so we stick to that overall format.
        """
        if self.inflation_indices_are_irrelevant:
            relevant_nd_array = np.take(self.as_ndarray, [0, -2, -1], axis=1)
        else:
            relevant_nd_array = self.as_ndarray
        return to_tuple_of_tuples(np.expand_dims(relevant_nd_array, axis=0))
        # return frozenset({(to_tuple_of_tuples(relevant_nd_array), 1)})


    def __hash__(self):
        # return hash(tuple(sorted([to_tuple_of_tuples(self.to_representative_function(factor)) for factor in self.factors])))
        return hash(self.signature)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.signature == other.signature
        elif isinstance(other, CompoundMonomial):
            return (other.nof_factors == 1) and self.__eq__(other.factors_as_atomic_monomials[0])
        else:
            assert isinstance(other,
                              self.__class__), f"Expected {self.__class__}, recieved {other} of {type(other)}{list(map(type, other))}."
            return False

    def __lt__(self, other):
        return self.signature < other.signature

    def __str__(self):
        # If a human readable name is available, we use it.
        if self.not_yet_named:
            return np.array2string(self.as_ndarray)
        else:
            return self.name

    def __repr__(self):
        return self.__str__()

    def update_rectified_array_based_on_fake_setting_correction(self, fake_setting_correction_func):
        if self.knowable_q and self.not_yet_fake_setting_rectified:
            self.rectified_ndarray = fake_setting_correction_func(self.rectified_ndarray)
            # self.rectified_ndarray_as_tuples = to_tuple_of_tuples(self.rectified_ndarray)
            self.not_yet_fake_setting_rectified = False


    def update_hash_via_to_representative_function(self, to_representative_function):
        # If to_representative has been called already, don't bother canonicalizing again.
        if self.not_yet_updated_by_to_representative:
            self.as_ndarray = to_representative_function(self.as_ndarray)
            # if self.inflation_indices_are_irrelevant:
            #     self.as_tuples = to_tuple_of_tuples(np.take(self.as_ndarray, [0, -2, -1], axis=1))
            # else:
            #     self.as_tuples = to_tuple_of_tuples(self.as_ndarray)
            # self.as_tuples = to_tuple_of_tuples(self.rectified_ndarray)
            # self.signature = self.as_tuples
            self.not_yet_updated_by_to_representative = False



    def update_name_and_symbol_given_observed_names(self, observable_names):
        if self.not_yet_named:
            self.name = atomic_monomial_to_name(observable_names=observable_names,
                                                atom=self.rectified_ndarray,
                                                human_readable_over_machine_readable=True,
                                                do_conditional=self.do_conditional)
            self.machine_name = atomic_monomial_to_name(observable_names=observable_names,
                                                        atom=self.as_ndarray,
                                                        human_readable_over_machine_readable=False)
            self.symbol = symbol_from_atomic_name(self.name)
            self.machine_symbol = symbol_from_atomic_name(self.machine_name)
            self.not_yet_named = False

    def as_CompoundMonomial(self):
        return CompoundMonomial([self])

    def compute_marginal(self, prob_array):
        assert self.knowable_q, "Can't compute marginals of unknowable probabilities."
        if self.fake_setting_rectified_yet:
            warnings.warn(
                "Warning: attempting to compute probabilities from raw operators, no setting rectification has been performed.")
        return compute_marginal(prob_array=prob_array,
                                atom=self.rectified_ndarray)


class CompoundMonomialMeta(type):
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls.cache = dict()

    def __call__(cls, list_of_atomic_monomials: List[AtomicMonomial]):
        quick_key = tuple(sorted(list_of_atomic_monomials))
        if quick_key not in cls.cache:
            # print('New Instance')
            cls.cache[quick_key] = super().__call__(list_of_atomic_monomials)
        return cls.cache[quick_key]


class CompoundMonomial(metaclass=CompoundMonomialMeta):

    # TODO: Add more constructors, for automatic monomial sanitization.
    @classmethod
    def from_Monomial(cls,
                      array2d: Union[np.ndarray, Tuple[Tuple[int]], List[List[int]]],
                      atomic_is_knowable=is_knowable,
                      sandwich_positivity=False,
                      idx=0):
        """
        This class is the only class the user should ever interact with. CompoundMonomial and AtomicMonomial are for internal use only.
        """
        # self.to_representative = lambda mon: tuple(tuple(vec) for vec in to_representative(mon))

        as_ndarray = np.asarray(array2d)
        if len(as_ndarray):
            assert as_ndarray.ndim == 2, 'Expected 2 dimension numpy array instead of ' + np.array2string(as_ndarray)
        else:
            as_ndarray = np.empty((0, 3), dtype=np.uint8)
        n_ops, op_length = as_ndarray.shape
        assert op_length >= 3, 'Expected at least 3 digits to specify party, outcome, settings.'

        _factors = factorize_monomial(as_ndarray, canonical_order=False)
        obj = cls(list_of_atomic_monomials=[AtomicMonomial(factor,
                                                           atomic_is_knowable=atomic_is_knowable,
                                                           sandwich_positivity=sandwich_positivity)
                                            for factor in _factors])
        obj.idx = idx
        obj.as_ndarray = as_ndarray
        # if idx > 1:
        #     obj.known_status = 'No'
        # else:
        #     obj.known_status = 'Yes'
        # # self.unknown_part = self.as_ndarray
        # obj.known_value = 1.
        # obj.unknown_signature = obj.signature
        return obj

    __slots__ = ['as_ndarray',
                 'n_ops',
                 'op_length',
                 # 'as_tuples',
                 # 'is_physical',
                 '_factors',
                 'factors_as_atomic_monomials',
                 # 'signature',
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
                 # 'known_status',
                 # 'known_value',
                 # 'unknown_part', # Deprecated by new 'unknown_signature'
                 # 'representative', # Deprecated by 'signature'
                 # 'unknown_signature',
                 'inflation_indices_are_irrelevant',
                 # 'factors_with_irrelevant_copy_indices',
                 'factors_with_relevant_copy_indices',
                 '_factors_repr',  # Emi to Elie: See comment the 'set_values' method, to be deprecated
                 # 'knowable_factors_names',
                 # 'knowable_part_name',
                 # 'knowable_part_machine_readable_name',
                 # 'unknowable_part_name',
                 # 'unknowable_part_machine_readable_name',
                 # '_names_of_factors',
                 # '_symbols_of_factors',
                 # '_machine_readable_names_of_factors',
                 # 'name',
                 'machine_readable_name',
                 # 'symbol',
                 'machine_readable_symbol',
                 ]

    def __init__(self, list_of_atomic_monomials: List[AtomicMonomial]):
        """
        This class is incredibly inefficient unless knowable_q has built-in memoization.
        It is designed to categorize monomials into known, semiknown, unknown, etc.
        Note that if knowable_q changes (such as given partial information) we can update this on the fly.
        """
        self.factors_as_atomic_monomials = sorted(list_of_atomic_monomials)
        self.nof_factors = len(self.factors_as_atomic_monomials)
        self.is_atomic = (self.nof_factors <= 1)
        # self.signature = tuple(sorted(self.factors_as_atomic_monomials))
        # if self.nof_factors == 1:
        #     self.signature = self.factors_as_atomic_monomials[0]  # So as to match atomic signature
        # else:
        #     self.signature = tuple(sorted(self.factors_as_atomic_monomials))
        # self.unknown_signature = self.signature

        self.knowable_factors = tuple(factor for factor in self.factors_as_atomic_monomials if factor.knowable_q)
        self.unknowable_factors = tuple(factor for factor in self.factors_as_atomic_monomials if not factor.knowable_q)

        self.factors_with_relevant_copy_indices = tuple(
            factor for factor in self.factors_as_atomic_monomials if not factor.inflation_indices_are_irrelevant)

        self.knowable_q = all(factor.knowable_q for factor in self.factors_as_atomic_monomials)
        self.physical_q = all(factor.physical_q for factor in self.factors_as_atomic_monomials)
        self._unknowable_block_len = sum(factor.n_ops for factor in self.unknowable_factors)
        self.n_ops = sum(factor.n_ops for factor in self.factors_as_atomic_monomials)

        if self._unknowable_block_len == 0:
            self.knowability_status = 'Yes'
        elif self._unknowable_block_len == self.n_ops:
            self.knowability_status = 'No'
        else:
            self.knowability_status = 'Semi'

    def __str__(self):
        # If a human readable name is available, we use it.
        try:
            return self.name
        except AttributeError:
            return np.array2string(self.as_ndarray)

    def __repr__(self):
        return self.__str__()

    @property
    def as_counter(self):
        return Counter(self.factors_as_atomic_monomials)

    def __len__(self):
        return self.nof_factors

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.as_counter == other.as_counter
        elif isinstance(other, AtomicMonomial):
            return (self.nof_factors == 1) and other.__eq__(self.factors_as_atomic_monomials[0])
        else:
            assert isinstance(other, self.__class__), f"Expected {self.__class__}, recieved {other} of {type(other)}{list(map(type, other))}."
            return False

    @property
    def signature(self):
        # if self.nof_factors == 1:
        #     return self.factors_as_atomic_monomials[0]  # So as to match atomic signature
        # else:
        return tuple(sorted(self.factors_as_atomic_monomials))
        # return frozenset(self.as_counter.items())

    def __hash__(self):
        # return hash(tuple(sorted([to_tuple_of_tuples(self.to_representative_function(factor)) for factor in self.factors])))
        return hash(self.signature)


    # Elie note to Emi: This should eventually not be neccessary, if we call the update outcome of the Monomial class.
    def update_atomic_constituents(self, to_representative_function, just_inflation_indices=False):
        if not just_inflation_indices:
            for atomic_mon in self.factors_as_atomic_monomials:
                atomic_mon.update_hash_via_to_representative_function(to_representative_function)
        else:
            for atomic_mon in self.factors_with_relevant_copy_indices:
                atomic_mon.update_hash_via_to_representative_function(
                    to_representative_function)  # Automatically changes self.factors_as_atomic_monomials
        # if self.nof_factors == 1:
        #     self.signature = self.factors_as_atomic_monomials[0]
        # else:
        #     self.signature = tuple(sorted(self.factors_as_atomic_monomials))

    def update_rectified_arrays_based_on_fake_setting_correction(self, fake_setting_correction_func):
        for atomic_mon in self.knowable_factors:
            atomic_mon.update_rectified_array_based_on_fake_setting_correction(fake_setting_correction_func)

    # def update_hash_via_to_representative_function(self, to_representative_function):
    #     # self.factors_with_relevant_copy_indices = tuple(to_representative_function(factor) for factor in self.factors_with_relevant_copy_indices)
    #     # # WARNING: WE CANNOT STACK THE FACTORS AFTER CONVERTING TO REPRESENTATIVE!
    #     # # The alternative is to call to_representative_function on the entire as_ndarray, instead of on the individual factors.
    #     # self.as_tuples = self.factors_with_irrelevant_copy_indices + tuple(sorted(
    #     #     to_tuple_of_tuples(factor) for factor in self.factors_with_relevant_copy_indices))
    #     self.as_ndarray = to_representative_function(self.as_ndarray)
    #     self.as_tuples = to_tuple_of_tuples(self.as_ndarray)


        # return self.signature == other.signature

    # def __lt__(self, other):
    #     return self.signature < other.signature

    # def factors_as_block(self, factors): #DEPRECATED NEED!
    #     if len(factors):
    #         return np.concatenate(factors)
    #     else:
    #         return np.empty((0, self.op_length), dtype=np.uint8)

    def evaluate_given_valuation_of_knowable_part(self, valuation_of_knowable_part, use_lpi_constraints=True):
        actually_known_factors = np.logical_not(np.isnan(valuation_of_knowable_part))
        known_value = float(np.prod(np.compress(
            actually_known_factors,
            valuation_of_knowable_part)))
        # nof_known_factors = np.count_nonzero(actually_known_factors)
        unknown_factors = [factor for factor, known in
                           zip(self.knowable_factors,
                               actually_known_factors)
                           if not known]
        unknown_factors.extend(self.unknowable_factors)
        # raw_unknown_signature = tuple(sorted(knowable_factors_which_are_not_known.extend(self.unknowable_factors)))
        # self.unknown_part = np.concatenate((
        #     self.factors_as_block(knowable_factors_which_are_not_known),
        #     self.unknowable_factors_as_block))
        unknown_len = len(unknown_factors)
        if unknown_len == 0 or (np.isclose(known_value, 0) and use_lpi_constraints):
            known_status = 'Yes'
        elif unknown_len == self.n_ops or (not use_lpi_constraints):
            known_status = 'No'
        else:
            known_status = 'Semi'
        return known_value, CompoundMonomial(unknown_factors), known_status
        # if unknown_len == 1:
        #     self.unknown_signature = raw_unknown_signature[0] #treat atomic cases special
        # else:
        #     self.unknown_signature = raw_unknown_signature

    # TODO: Function is WORK IN PROGRESS! Should this be a dict?
    def evaluate_given_atomic_monomials_dict(self, dict_of_known_atomic_monomials: Dict[AtomicMonomial, float], use_lpi_constraints=True):
        "Yields both a numeric value and a CompoundMonomial corresponding to the unknown part."
        known_value = 1.
        unknown_factors = Counter()
        for factor, power in self.as_counter.items():
            temp_value = dict_of_known_atomic_monomials.get(factor, np.nan)
            if np.isnan(temp_value):
                unknown_factors[factor] = power
            else:
                known_value *= (temp_value ** power)
        #
        # valuation_of_knowable_part = [dict_of_known_atomic_monomials.get(atomic_mon, np.nan) for atomic_mon in
        #                               self.factors_as_atomic_monomials]
        # actually_known_factors = np.logical_not(np.isnan(valuation_of_knowable_part))
        # known_value = float(np.prod(np.compress(
        #     actually_known_factors,
        #     valuation_of_knowable_part)))
        # unknown_factors = [factor for factor, known in
        #                    zip(self.factors_as_atomic_monomials,
        #                        actually_known_factors)
        #                    if not known]
        unknown_len = len(unknown_factors)
        if unknown_len == 0 or (np.isclose(known_value, 0) and use_lpi_constraints):
            known_status = 'Yes'
        elif unknown_len == self.n_ops or (not use_lpi_constraints):
            known_status = 'No'
        else:
            known_status = 'Semi'
        return known_value, CompoundMonomial(unknown_factors.elements()), known_status

    def update_name_and_symbol_given_observed_names(self, observable_names):
        for factor in self.factors_as_atomic_monomials:
            factor.update_name_and_symbol_given_observed_names(observable_names)
        self.machine_readable_name = atomic_monomial_to_name(observable_names=observable_names,
                                                             atom=self.as_ndarray,
                                                             human_readable_over_machine_readable=False)
        self.machine_readable_symbol = (sympy.S.One if self.machine_readable_name == '1' else
                                        np.prod([sympy.Symbol(op, commutative=False)
                                                 for op in self.machine_readable_name.split('*')]))

    @property
    def _names_of_factors(self):
        if not any(factor.not_yet_named for factor in self.factors_as_atomic_monomials):
            return sorted(factor.name for factor in self.knowable_factors) + sorted(
                factor.name for factor in self.unknowable_factors)
        else:
            raise AttributeError

    @property
    def _symbols_of_factors(self):
        if not any(factor.not_yet_named for factor in self.factors_as_atomic_monomials):
            return sorted(factor.symbol for factor in self.knowable_factors) + sorted(
                factor.symbol for factor in self.unknowable_factors)
        else:
            raise AttributeError

    @property
    def name(self):
        return name_from_atomic_names(self._names_of_factors)

    @property
    def symbol(self):
        return symbol_from_atomic_symbols(self._symbols_of_factors)

    # TODO: Implement machine_readable names and symbols?

    # def update_given_prob_dist(self, prob_array):
    #     if prob_array is None:
    #         self.known_status = 'No'
    #         self.known_value = 1
    #         self.unknown_signature = self.signature
    #         # self.unknown_part = self.as_ndarray
    #     else:
    #         hashable_prob_array = to_tuple_of_tuples(prob_array)
    #         valuation_of_knowable_part = np.array([
    #             compute_marginal_memoized(hashable_prob_array, atom.as_tuples) #Note that atom.as_tuples uses the RECTIFIED ndarray
    #             for atom in self.knowable_factors])
    #         return self.update_given_valuation_of_knowable_part(valuation_of_knowable_part)

    def to_symbol(self, objective_compatible=False):
        if objective_compatible:
            return self.machine_readable_symbol
        else:
            return self.symbol

    def compute_marginal(self, prob_array):
        assert self.knowable_q, "Can't compute marginals of unknowable probabilities."
        v = 1.
        for factor, power in self.as_counter.items():
            v *= (compute_marginal(prob_array=prob_array,
                                  atom=factor.rectified_ndarray) ** power)
        return v


def Monomial(*args, **kwargs):
    return CompoundMonomial.from_Monomial(*args, **kwargs)
