# from __future__ import annotations
# from __future__ import absolute_import
# from __future__ import with_statement
import numpy as np
from causalinflation.quantum.general_tools import factorize_monomial, is_physical, is_knowable  # to_representative_aux
# import itertools
# from functools import cached_property
from functools import lru_cache

from typing import List, Tuple, Union
from collections.abc import Iterable
# ListOrTuple = NewType("ListOrTuple", Union[List, Tuple])
# MonomInputType = NewType("NumpyCompat", Union[np.ndarray, ListOrTuple[ListOrTuple[int]]])


def to_tuple_of_tuples(monomial: np.ndarray) -> tuple:
    if isinstance(monomial, tuple):
        return monomial
    elif isinstance(monomial, np.ndarray):
        if monomial.ndim >= 2:
            return tuple(to_tuple_of_tuples(operator) for operator in monomial)
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
        participating_parties = atom[:,  0]
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


class Monomial(object):
    __slots__ = ['as_ndarray',
                 'n_ops',
                 'op_length',
                 'as_tuples',
                 'is_physical',
                 'factors',
                 'nof_factors',
                 'atomic_knowability_status',
                 'knowable_factors_uncompressed',
                 'knowable_factors',
                 'knowable_factors_names',
                 'knowable_part_name',
                 'unknowable_factors',
                 'unknowable_factors_as_block',
                 'unknown_part_name',
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
                 'name'
                 ]

    def __init__(self, array2d: Union[np.ndarray, Tuple[Tuple[int]], List[List[int]]],
                 atomic_is_knowable=is_knowable,
                 sandwich_positivity=False, idx=0):
        """
        This class is incredibly inefficient unless knowable_q has built-in memoization.
        It is designed to categorize monomials into known, semiknown, unknown, etc.
        Note that if knowable_q changes (such as given partial information) we can update this on the fly.
        """
        # self.to_representative = lambda mon: tuple(tuple(vec) for vec in to_representative(mon))

        self.idx = idx
        self.as_ndarray = np.asarray(array2d)
        assert self.as_ndarray.ndim == 2, 'Expected 2 dimension numpy array.'
        self.n_ops, self.op_length = self.as_ndarray.shape
        assert self.op_length >= 3, 'Expected at least 3 digits to specify party, outcome, settings.'
        self.as_tuples = to_tuple_of_tuples(self.as_ndarray)
        self.factors = factorize_monomial(self.as_ndarray, canonical_order=False)
        self.nof_factors = len(self.factors)
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
        self.knowable_q = (len(self.knowable_factors) == self.nof_factors)
        self.physical_q = self.knowable_q or is_physical(self.unknowable_factors_as_block,
                                                         sandwich_positivity=sandwich_positivity)
        self._unknowable_block_len = len(self.unknowable_factors_as_block)

        if self._unknowable_block_len == 0:
            self.knowability_status = 'Yes'
        elif self._unknowable_block_len == self.n_ops:
            self.knowability_status = 'No'
        else:
            self.knowability_status = 'Semi'

        self.known_status = 'No'
        self.unknown_part = self.unknowable_factors_as_block
        self.known_value = 1.

    def __str__(self):
        return np.array2string(self.as_ndarray)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.as_tuples)

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
