import numpy as np

from collections import Counter
from functools import total_ordering
from typing import Tuple, Dict

from .fast_npa import mon_is_zero
from .general_tools import is_physical
from .monomial_utils import (compute_marginal,
                             name_from_atomic_names,
                             symbol_from_atomic_name,
                             symbol_prod)

@total_ordering
class InternalAtomicMonomial(object):
    __slots__ = ["as_ndarray",
                 "do_conditional",
                 "is_one",
                 "is_zero",
                 "is_knowable",
                 "n_ops",
                 "op_length",
                 "rectified_ndarray",
                 "sdp"
                 ]

    def __init__(self, inflation_sdp_instance, array2d: np.ndarray):
        """
        This uses methods from the InflationSDP instance, and so must be constructed with that passed as first argument.

        DOCUMENTATION NEEDED: What is this object, what and where it is used for, and what it does.
        """
        self.sdp        = inflation_sdp_instance
        self.as_ndarray = np.asarray(array2d, dtype=self.sdp.np_dtype)
        self.n_ops, self.op_length = self.as_ndarray.shape
        assert self.op_length == self.sdp._nr_properties, \
            ("An AtomicMonomial should be a 2-d array where each row is a list"
             + f" of integers of length {self.sdp._nr_properties}. The first "
             + "index corresponds to the party, the last one to the outcome, "
             + "the second-to-last to the setting, and the rest to the "
             + "inflation copies.")
        self.is_zero = mon_is_zero(self.as_ndarray)
        self.is_one  = (self.n_ops == 0)
        self.is_knowable = (self.is_zero
                            or self.is_one
                            or self.sdp.atomic_knowable_q(self.as_ndarray))
        self.do_conditional = False
        # Save also array with the original setting, not just the effective one
        if self.is_knowable:
            self.rectified_ndarray = np.asarray(
                self.sdp.rectify_fake_setting(np.take(self.as_ndarray,
                                                      [0, -2, -1],
                                                      axis=1)),
                                                dtype=int)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        for attr in self.__slots__:
            try:
                result.__setattr__(attr, self.__getattribute__(attr))
            except AttributeError:
                pass
        return result

    def __eq__(self, other):
        """Whether the Monomial is equal to the ``other`` Monomial."""
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        """Return the hash of the Monomial."""
        return hash(self.signature)

    def __lt__(self, other):
        """Whether the Monomial is lexicographically smaller than the ``other``
        Monomial.
        """
        return self.signature < other.signature

    def __repr__(self):
        """Return the name of the Monomial."""
        return self.__str__()

    def __str__(self):
        """Return the name of the Monomial."""
        return self.name

    @property
    def dagger(self):
        """Returns the adjoint of the Monomial."""
        conjugate_ndarray   = self.sdp.conjugate_ndarray(self.as_ndarray)
        conjugate_signature = self.sdp.from_2dndarray(conjugate_ndarray)
        if conjugate_signature != self.signature:
            dagger = self.__copy__()
            dagger.as_ndarray = conjugate_ndarray
            return dagger
        else:
            return self

    @property
    def is_physical(self):
        """Whether the expectation value of the monomial is non-negative for
        any quantum state
        """
        return self.is_knowable or is_physical(self.as_ndarray)

    @property
    def name(self):
        """A string representing the monomial. In case of knowable monomials, it
        is of the form p(outputs|inputs). Otherwise it represents the
        expectation value of the monomial with bracket notation.
        """
        if self.is_one:
            return "1"
        elif self.is_zero:
            return "0"
        elif self.is_knowable:
            # Use notation p(outputs|settings)
            # Convention in numpy monomial format is first party = 1
            party_indices = self.rectified_ndarray[:, 0] - 1
            parties       = np.take(self.sdp.names, party_indices.tolist())
            inputs  = [str(input) for input in self.rectified_ndarray[:, -2]]
            outputs = [str(output) for output in self.rectified_ndarray[:, -1]]
            p_divider = "" if all(len(p) == 1 for p in parties) else ","
            # We will probably never have more than 1 digit cardinalities, but who knows...
            i_divider = "" if all(len(i) == 1 for i in inputs) else ","
            o_divider = "" if all(len(o) == 1 for o in outputs) else ","
            if self.do_conditional:
                return ("p" + p_divider.join(parties) +
                        "(" + o_divider.join(outputs) +
                        " do: " + i_divider.join(inputs) + ")")
            else:
                return ("p" + p_divider.join(parties) +
                        "(" + o_divider.join(outputs) +
                        "|" + i_divider.join(inputs) + ")")
        else:
            # Use expectation value notation
            operators = []
            for op in self.as_ndarray:
                operators.append("_".join([self.sdp.names[op[0] - 1]]
                                          + [str(i) for i in op[1:]]))
            return "<" + " ".join(operators_as_strings) + ">"

    @property
    def signature(self):
        return self.sdp.from_2dndarray(self.as_ndarray)

    @property
    def symbol(self):
        """Return a sympy Symbol representing the monomial."""
        return symbol_from_atomic_name(self.name)

    def compute_marginal(self, prob_array):
        """Given a probability distribution, compute the value of the Monomial.

        Parameters
        ----------
        prob_array : numpy.ndarray
            The target probability distribution. The dimensions of this array
            are (*outcomes_per_party, *settings_per_party), where the settings
            are explicitly 1 is there is only one measurement performed.

        Returns
        -------
        float
            The value of the corresponding probability (which can be a marginal)
        """
        if self.is_zero:
            return 0.
        else:
            return compute_marginal(prob_array, self.rectified_ndarray)


class CompoundMonomial(object):
    __slots__ = ["factors",
                 "is_atomic",
                 "is_zero",
                 "is_one",
                 "n_factors",
                 "knowable_factors",
                 "unknowable_factors",
                 "n_knowable_factors",
                 "n_unknowable_factors",
                 "knowability_status",
                 "is_knowable",
                 "idx",
                 "mask_matrix"
                 ]

    def __init__(self, tuple_of_atomic_monomials: Tuple[InternalAtomicMonomial]):
        """
        This class is designed to categorize monomials into known, semiknown, unknown, etc.
        It also computes names for expectation values, and provides the ability to compare (in)equivalence.

        DOCUMENTATION NEEDED. What is this object, what is the input, what it is supposed to do.
        """
        default_factors    = tuple(sorted(tuple_of_atomic_monomials))
        conjugate_factors  = tuple(sorted(factor.dagger for factor in tuple_of_atomic_monomials))
        self.factors       = min(default_factors, conjugate_factors)
        self.n_factors     = len(self.factors)
        self.is_atomic     = (self.n_factors <= 1)
        self.is_knowable   = all(factor.is_knowable for factor in self.factors)
        knowable_factors   = []
        unknowable_factors = []
        for factor in self.factors:
            if factor.is_knowable:
                knowable_factors.append(factor)
            else:
                unknowable_factors.append(factor)
        self.knowable_factors     = tuple(knowable_factors)
        self.unknowable_factors   = tuple(unknowable_factors)
        self.n_knowable_factors   = len(self.knowable_factors)
        self.n_unknowable_factors = len(self.unknowable_factors)
        if self.n_unknowable_factors == 0:
            self.knowability_status = "Yes"
        elif self.n_unknowable_factors == self.n_factors:
            self.knowability_status = "No"
        else:
            self.knowability_status = "Semi"
        self.is_zero = any(factor.is_zero for factor in self.factors)
        self.is_one  = (all(factor.is_one for factor in self.factors)
                        or (self.n_factors == 0))

    def __eq__(self, other):
        """Whether the Monomial is equal to the ``other`` Monomial."""
        if isinstance(other, self.__class__):
            return self.as_counter == other.as_counter
        elif isinstance(other, InternalAtomicMonomial):
            return (self.n_factors == 1) and other.__eq__(self.factors[0])
        else:
            assert isinstance(other, self.__class__), \
                (f"Expected {self.__class__}, received {other} of " +
                 f"{type(other)}{list(map(type, other))}.")
            return False

    def __hash__(self):
        """Return the hash of the Monomial."""
        return hash(self.signature)

    def __len__(self):
        """Return the number of AtomicMonomials in the CompoundMonomial."""
        return self.n_factors

    def __repr__(self):
        """Return the name of the Monomial."""
        return self.__str__()

    def __str__(self):
        """Return the name of the Monomial."""
        return self.name

    @property
    def as_counter(self):
        """DOCUMENTATION NEEDED."""
        return Counter(self.factors)

    @property
    def n_ops(self):
        """DOCUMENTATION NEEDED."""
        return sum(factor.n_ops for factor in self.factors)

    @property
    def is_physical(self):
        """DOCUMENTATION NEEDED."""
        return all(factor.is_physical for factor in self.factors)

    @property
    def signature(self):
        """DOCUMENTATION NEEDED."""
        return tuple(sorted(self.factors))

    @property
    def name(self):
        """DOCUMENTATION NEEDED."""
        return name_from_atomic_names(self._names_of_factors)

    @property
    def symbol(self):
        """DOCUMENTATION NEEDED."""
        return symbol_prod(self._symbols_of_factors)

    @property
    def _names_of_factors(self):
        """DOCUMENTATION NEEDED."""
        return [factor.name for factor in self.factors]

    @property
    def _symbols_of_factors(self):
        """DOCUMENTATION NEEDED."""
        return [factor.symbol for factor in self.factors]

    def evaluate_given_valuation_of_knowable_part(self,
                                                  valuation_of_knowable_part,
                                                  use_lpi_constraints=True):
        """DOCUMENTATION NEEDED."""
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
            known_status = "Yes"
        elif unknown_len == self.n_factors or (not use_lpi_constraints):
            known_status = "No"
        else:
            known_status = "Semi"
        return known_value, unknown_factors, known_status


    def evaluate_given_atomic_monomials_dict(self,
                                             dict_of_known_atomic_monomials: Dict[InternalAtomicMonomial, float],
                                             use_lpi_constraints=True):
        """Yields both a numeric value and a CompoundMonomial corresponding to the unknown part.
        DOCUMENTATION NEEDED."""
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
            known_status = "Yes"
        elif unknown_len == self.n_factors or (not use_lpi_constraints):
            known_status = "No"
        else:
            known_status = "Semi"
        return known_value, unknown_factors, known_status

    def compute_marginal(self, prob_array):
        """DOCUMENTATION NEEDED."""
        assert self.is_knowable, "Can't compute marginals of unknowable probabilities."
        v = 1.
        for factor, power in self.as_counter.items():
            v *= (factor.compute_marginal(prob_array) ** power)
        return v

    def attach_idx_to_mon(self, idx: int):
        """DOCUMENTATION NEEDED."""
        if idx >= 0:
            self.idx = idx
