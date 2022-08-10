# from __future__ import annotations
# from __future__ import absolute_import
# from __future__ import with_statement
import numpy as np
from causalinflation.quantum.general_tools import factorize_monomial, is_physical, is_knowable  # to_representative_aux
# import itertools
from functools import cached_property

from typing import List, Tuple, Union
# ListOrTuple = NewType("ListOrTuple", Union[List, Tuple])
# MonomInputType = NewType("NumpyCompat", Union[np.ndarray, ListOrTuple[ListOrTuple[int]]])


class Monomial(object):
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
        assert self.as_ndarray.ndim == 2, 'Expected 2 dimension numpy array.'
        self.n_ops, self.op_length = self.as_ndarray.shape
        assert self.op_length >= 3, 'Expected at least 3 digits to specify party, outcome, settings.'
        self.as_tuples = tuple(tuple(vec) for vec in self.as_ndarray)
        self.atomic_is_knowable = atomic_is_knowable

        self.is_physical = lambda mon: is_physical(mon, sandwich_positivity=sandwich_positivity)

    # @cached_property
    # def as_representative(self):
    #     return tuple(tuple(vec) for vec in self.to_representative(self.as_ndarray))

    def atomic_is_not_knowable(self, mon):
        return not self.atomic_is_knowable(mon)

    def __str__(self):
        return np.array2string(self.as_ndarray)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.as_tuples)

    @cached_property
    def factors(self):
        return factorize_monomial(self.as_ndarray)

    def factors_as_block(self, factors):
        if len(factors):
            return np.concatenate(factors)
        else:
            return np.empty((0, self.op_length), dtype=np.int8)

    @cached_property
    def nof_factors(self):
        return len(self.factors)

    @cached_property
    def knowable_factors(self):
        # return [tuple(tuple(vec) for vec in np.take(factor, [0, -2, -1], axis=1))
        #         for factor in self.factors if self.atomic_is_knowable(factor)]
        return [tuple(tuple(vec) for vec in np.take(factor, [0, -2, -1], axis=1))
                for factor in self.knowable_factors_uncompressed]

    @cached_property
    def knowable_factors_uncompressed(self):
        return tuple(filter(self.atomic_is_knowable, self.factors))

    @cached_property
    def unknowable_factors(self):
        return tuple(filter(self.atomic_is_not_knowable, self.factors))

    # @cached_property
    # def unknowable_part(self):
    #     if len(self.unknowable_factors):
    #         return self.to_representative(np.vstack(self.unknowable_factors))
    #     else:
    #         return tuple()
        # if len(raw_unknowns) < 2:
        #     if len(raw_unknowns) == 1:
        #         return self.to_representative(raw_unknowns[0])
        #     else:
        #         return tuple()
        # else:
        #     candidate_unknown_part_representatives = []
        #     for perm in itertools.permutations(raw_unknowns, len(raw_unknowns)):
        #         candidate_unknown_part_representatives.append(self.to_representative(np.vstack(perm)))
        #     assert len(set(
        #         tuple(tuple(vec) for vec in monomial) for monomial in candidate_unknown_part_representatives)) <= 1, \
        #         "The 'to_representative' function is incorrectly sensitive to factor order."
        #     return sorted(candidate_unknown_part_representatives)[0]

    @cached_property
    def knowability_status(self):
        if len(self.knowable_factors) == self.nof_factors:
            return 'Yes'
        elif len(self.knowable_factors) > 0:
            return 'Semi'
        else:
            return 'No'

    @cached_property
    def knowable_q(self):
        return self.knowability_status == 'Yes'

    @cached_property
    def physical_q(self):
        if self.knowable_q:
            return True
        else:
            return self.is_physical(np.concatenate(self.unknowable_factors))

    # def decomposition_given_dist(self, knowable_atoms_dict):
    #     yield_nan = lambda x: np.nan
    #     error_free_dict = defaultdict(yield_nan, knowable_atoms_dict)
    #     #We want to know the values we can obtain.
    #
    #     return np.prod([knowable_atoms_dict[knowable_atom] for knowable_atom in error_free_dict])

#
#
#
# class AtomicMonomial(object):
#     def __init__(self, array2d:np.ndarray, knowable_q, to_representative):
#         self.knowable = knowable_q(array2d)
#         if self.knowable:
#             self.representative =
#             return AtomicMonomial
#
#         self.as_ndarray = np.asarray(array1d)
#         assert self.as_ndarray.ndim == 1, 'Expected 2 dimension numpy array.'
#         self.n_ops, self.op_length = self.as_ndarray.shape
#         self.as_tuple = tuple(self.as_ndarray)
#
#
#
#     def __str__(self):
#         return np.array2string(self.as_ndarray)
#
#     def __repr__(self):
#         return self.as_tuple
#
