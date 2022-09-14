import numpy as np
from causalinflation.quantum.general_tools import is_physical
from functools import total_ordering
from typing import Tuple, Dict
from collections import Counter

from causalinflation.quantum.monomial_utils import (
    compute_marginal, symbol_from_atomic_name, name_from_atomic_names, symbol_from_atomic_symbols)

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
                 'is_zero',
                 'is_one'
                 ]

    def __init__(self, inflation_sdp_instance, array2d: np.ndarray):
        """
        This uses methods from the InflationSDP instance, and so must be constructed with that passed as first argument.
        """
        self.sdp = inflation_sdp_instance
        self.as_ndarray = np.asarray(array2d, dtype=self.sdp.np_dtype)
        self.n_ops, self.op_length = self.as_ndarray.shape
        assert self.op_length == self.sdp._nr_properties, "We insist on well-formed 2d arrays as input to AtomicMonomial."
        self.is_zero = np.any(np.logical_not(self.as_ndarray[:, 0]))  # Party indexing starts at 1, so a zero in the zero slot indicates bad.
        self.is_one = (self.n_ops == 0)
        self.knowable_q = self.is_zero or self.is_one or self.sdp.atomic_knowable_q(self.as_ndarray)
        self.do_conditional = False
        if self.knowable_q:
            self.rectified_ndarray = np.asarray(self.sdp.rectify_fake_setting_atomic_factor(np.take(self.as_ndarray, [0, -2, -1], axis=1)), dtype=int)


    @property
    def physical_q(self):
        return self.knowable_q or is_physical(self.as_ndarray)

    @property
    def signature(self):
        return self.sdp.from_2dndarray(self.as_ndarray)


    def __hash__(self):
        return hash(self.signature)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__


    def __lt__(self, other):
        return self.signature < other.signature

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


    @property
    def name(self):
        if self.is_one:
            return '1'
        elif self.is_zero:
            return '0'
        elif self.knowable_q:
            party_indices = self.rectified_ndarray[:, 0] - 1
            parties = np.take(self.sdp.names, party_indices.tolist())  # Convention in numpy monomial format is first party = 1
            inputs = [str(input) for input in self.rectified_ndarray[:, -2].tolist()]
            outputs = [str(output) for output in self.rectified_ndarray[:, -1].tolist()]
            p_divider = '' if all(len(p) == 1 for p in parties) else ','
            # We will probably never have more than 1 digit cardinalities, but who knows...
            i_divider = '' if all(len(i) == 1 for i in inputs) else ','
            o_divider = '' if all(len(o) == 1 for o in outputs) else ','
            if self.do_conditional:
                return ('p' + p_divider.join(parties) +
                        '(' + o_divider.join(outputs) + ' do: ' + i_divider.join(inputs) + ')')
            else:
                return ('p' + p_divider.join(parties) +
                        '(' + o_divider.join(outputs) + '|' + i_divider.join(inputs) + ')')
        else:
            operators_as_strings = []
            for op in self.as_ndarray.tolist():  # this handles the UNKNOWN factors.
                operators_as_strings.append('_'.join([self.sdp.names[op[0] - 1]]  # party idx
                                                     + [str(i) for i in op[1:]]))
            return 'OpSeq[' + ', '.join(operators_as_strings) + ']'

    @property
    def symbol(self):
        return symbol_from_atomic_name(self.name)

    def compute_marginal(self, prob_array):
        if self.is_zero:
            return 0.
        else:
            return compute_marginal(prob_array=prob_array,
                                    atom=self.rectified_ndarray)


class CompoundMonomial(object):
    __slots__ = ['factors_as_atomic_monomials',
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
        unknown_factors = list(unknown_factors_counter.elements())
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
            v *= (factor.compute_marginal(prob_array) ** power)
        return v

    def attach_idx_to_mon(self, idx: int):
        if idx >= 0:
            self.idx = idx
