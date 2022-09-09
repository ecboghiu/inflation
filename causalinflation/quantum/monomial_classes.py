# from __future__ import annotations
# from __future__ import absolute_import
# from __future__ import with_statement
# import warnings
import warnings

import numpy as np
from causalinflation.quantum.general_tools import is_physical
from functools import total_ordering
from typing import Tuple, Dict
from collections import Counter

from causalinflation.quantum.monomial_utils import (
    compute_marginal, atomic_monomial_to_name,
    symbol_from_atomic_name, name_from_atomic_names, symbol_from_atomic_symbols)

@total_ordering
class InternalAtomicMonomial(object):
    __slots__ = ['sdp',
                 'as_ndarray',
                 'rectified_ndarray',
                 'n_ops',
                 'op_length',
                 'inflation_indices_are_irrelevant',
                 'knowable_q',
                 'do_conditional',
                 # 'physical_q',
                 # 'name',
                 # 'symbol',
                 # 'machine_name',
                 # 'machine_symbol'
                 ]

    #TODO: Initialize using tuple_of_tuples, for better to_representative efficiency.
    def __init__(self, inflation_sdp_instance, array2d: np.ndarray):
        """
        This uses methods from the InflationSDP instance, and so must be constructed with that passed as first argument.
        """
        # assert isinstance(array2d, np.ndarray), 'We only accept numpy arrays for AtomicMonomial initialization.'
        self.sdp = inflation_sdp_instance
        self.as_ndarray = np.asarray(array2d, dtype=self.sdp.np_dtype)
        # self.as_ndarray = self.sdp.to_representative_ndarray(array2d) #Perhaps we will not need, if constructor is safe!
        self.n_ops, self.op_length = self.as_ndarray.shape
        assert self.op_length == self.sdp._nr_properties, "We "
        self.knowable_q = self.sdp.atomic_knowable_q(self.as_ndarray)
        self.do_conditional = False # (not self.knowable_q) and is_knowable(self.as_ndarray)  # TODO: Improve ability to handle do conditionals.
        if self.knowable_q:  # self.inflation_indices_are_irrelevant:
            self.rectified_ndarray = self.sdp.rectify_fake_setting_atomic_factor(np.take(self.as_ndarray, [0, -2, -1], axis=1))

    @property
    def physical_q(self):
        return is_physical(self.as_ndarray)

    @property
    def signature(self):
        return self.sdp.from_2dndarray(self.as_ndarray) #WILL USE tobytes and frombuffer


    def __hash__(self):
        # return hash(tuple(sorted([to_tuple_of_tuples(self.to_representative_function(factor)) for factor in self.factors])))
        return hash(self.signature)

    def __eq__(self, other):
        # return np.array_equal(self.as_ndarray, other.as_ndarray)
        return self.__hash__() == other.__hash__ #Maybe this improves performance somewhat?
        # if isinstance(other, self.__class__):
        #     return np.array_equal(self.as_ndarray, other.as_ndarray)
        # elif isinstance(other, CompoundMonomial):
        #     return (other.nof_factors == 1) and self.__eq__(other.factors_as_atomic_monomials[0])
        # else:
        #     assert isinstance(other,
        #                       self.__class__), f"Expected {self.__class__}, recieved {other} of {type(other)}{list(map(type, other))}."
        #     return False

    def __lt__(self, other):
        return self.signature < other.signature

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


    @property
    def name(self):
        if self.knowable_q: #TODO: or self.do_conditional, eventually
            return atomic_monomial_to_name(observable_names=self.sdp.names,
                                                atom=self.rectified_ndarray,
                                                human_readable_over_machine_readable=True,
                                                do_conditional=self.do_conditional)
        else:
            return atomic_monomial_to_name(observable_names=self.sdp.names,
                                                atom=self.as_ndarray,
                                                human_readable_over_machine_readable=True,
                                                do_conditional=self.do_conditional)
    @property
    def symbol(self):
        return symbol_from_atomic_name(self.name)

    def compute_marginal(self, prob_array):
        # assert self.knowable_q, "Can't compute marginals of unknowable probabilities."
        return compute_marginal(prob_array=prob_array,
                                atom=self.rectified_ndarray)



# class CompoundMonomial(metaclass=CompoundMonomialMeta):
class CompoundMonomial(object):
    __slots__ = [# 'as_ndarray',
                 'factors_as_atomic_monomials',
                 'is_atomic',
                 'nof_factors',
                 'knowable_factors',
                 'unknowable_factors',
                 'nof_knowable_factors',
                 'nof_unknowable_factors',
                 'knowability_status',
                 'knowable_q',
                 'idx',
                 'mask_matrix'
                 ]

    def __init__(self, tuple_of_atomic_monomials: Tuple[InternalAtomicMonomial]):
        """
        This class is incredibly inefficient unless knowable_q has built-in memoization.
        It is designed to categorize monomials into known, semiknown, unknown, etc.
        Note that if knowable_q changes (such as given partial information) we can update this on the fly.
        """
        self.factors_as_atomic_monomials = tuple_of_atomic_monomials
        self.nof_factors = len(self.factors_as_atomic_monomials)
        self.is_atomic = (self.nof_factors <= 1)
        # self.factors_with_relevant_copy_indices = tuple(
        #     factor for factor in self.factors_as_atomic_monomials if not factor.inflation_indices_are_irrelevant)
        # self.physical_q = all(factor.physical_q for factor in self.factors_as_atomic_monomials)
        self.knowable_q = all(factor.knowable_q for factor in self.factors_as_atomic_monomials)
        self.knowable_factors = tuple(factor for factor in self.factors_as_atomic_monomials if factor.knowable_q)
        self.unknowable_factors = tuple(factor for factor in self.factors_as_atomic_monomials if not factor.knowable_q)
        self.nof_knowable_factors = len(self.knowable_factors)
        self.nof_unknowable_factors = len(self.unknowable_factors)
        if self.nof_unknowable_factors == 0:
            self.knowability_status = 'Yes'
        elif self.nof_unknowable_factors == self.nof_factors:
            self.knowability_status = 'No'
        else:
            self.knowability_status = 'Semi'

    @property
    def physical_q(self):
        return all(factor.physical_q for factor in self.factors_as_atomic_monomials)

    def __str__(self):
        return self.name

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
        elif isinstance(other, InternalAtomicMonomial):
            return (self.nof_factors == 1) and other.__eq__(self.factors_as_atomic_monomials[0])
        else:
            assert isinstance(other, self.__class__), f"Expected {self.__class__}, recieved {other} of {type(other)}{list(map(type, other))}."
            return False

    @property
    def signature(self):
        return tuple(sorted(self.factors_as_atomic_monomials))

    def __hash__(self):
        return hash(self.signature)

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
        unknown_len = len(unknown_factors)
        if unknown_len == 0 or (np.isclose(known_value, 0) and use_lpi_constraints):
            known_status = 'Yes'
        elif unknown_len == self.nof_factors or (not use_lpi_constraints):
            known_status = 'No'
        else:
            known_status = 'Semi'
        return known_value, unknown_factors, known_status


    def evaluate_given_atomic_monomials_dict(self, dict_of_known_atomic_monomials: Dict[InternalAtomicMonomial, float], use_lpi_constraints=True):
        "Yields both a numeric value and a CompoundMonomial corresponding to the unknown part."
        known_value = 1.
        unknown_factors_counter = Counter()
        for factor, power in self.as_counter.items():
            temp_value = dict_of_known_atomic_monomials.get(factor, np.nan)
            if np.isnan(temp_value):
                unknown_factors_counter[factor] = power
            else:
                known_value *= (temp_value ** power)
        # warnings.warn(self.name + str(self.unknowable_factors))
        unknown_factors = list(unknown_factors_counter.elements())
        # unknown_len = unknown_factors_counter.total() #Since it is a counter. Only available in Python 3.10+
        unknown_len = len(unknown_factors)
        if unknown_len == 0 or (np.isclose(known_value, 0) and use_lpi_constraints):
            known_status = 'Yes'
        elif unknown_len == self.nof_factors or (not use_lpi_constraints):
            known_status = 'No'
        else:
            known_status = 'Semi'
        return known_value, unknown_factors, known_status

    @property
    def _names_of_factors(self):
        return [factor.name for factor in self.factors_as_atomic_monomials]

    @property
    def _symbols_of_factors(self):
        return [factor.symbol for factor in self.factors_as_atomic_monomials]

    @property
    def name(self):
        return name_from_atomic_names(self._names_of_factors)

    @property
    def symbol(self):
        return symbol_from_atomic_symbols(self._symbols_of_factors)

    def compute_marginal(self, prob_array):
        assert self.knowable_q, "Can't compute marginals of unknowable probabilities."
        v = 1.
        for factor, power in self.as_counter.items():
            v *= (compute_marginal(prob_array=prob_array,
                                   atom=factor.rectified_ndarray) ** power)
        return v

    def to_symbol(self, objective_compatible=False):
        if not objective_compatible:
            return self.symbol
        else:
            raise Exception("This method of calling machine-readable name from Monomial is no longer available.")
            # warnings.warn('This method is highly unreliable and should never be used!! In fact, as_ndarray may not be available.')
            # machine_readable_name = atomic_monomial_to_name(observable_names=self.sdp.names,
            #                                                 atom=self.as_ndarray,
            #                                                 human_readable_over_machine_readable=False)
            # machine_readable_symbol = (sympy.S.One if machine_readable_name == '1' else
            #                                 np.prod([sympy.Symbol(op, commutative=False)
            #                                          for op in machine_readable_name.split('*')]))
            # return machine_readable_symbol

    def attach_idx_to_mon(self, idx: int):
        if idx >= 0:
            self.idx = idx