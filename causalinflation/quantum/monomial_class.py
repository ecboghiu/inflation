import numpy as np
from general_tools import factorize_monomial, is_physical
import itertools
from functools import cached_property


class Monomial(object):
    def __init__(self, array2d:np.ndarray, knowable_q, to_representative, **kwargs):
        """
        This class is incredibly inefficient unless knowable_q has built-in memoization.
        It is designed to categorize monomials into known, semiknown, unknown, etc.
        Note that if knowable_q changes (such as given partial information) we can update this on the fly.
        """
        self.as_ndarray = np.asarray(array2d)
        assert self.as_ndarray.ndim == 2, 'Expected 2 dimension numpy array.'
        self.n_ops, self.op_length = self.as_ndarray.shape
        assert self.op_length >= 3, 'Expected at least 3 digits to specify party, outcome, settings.'
        self.as_tuples = tuple(tuple(vec) for vec in self.as_ndarray)
        self.knowable_q = knowable_q
        self.to_representative = to_representative
        self.is_physical = lambda mon: is_physical(mon, **kwargs)

    def unknowable_q(self, mon):
        return not self.knowable_q(mon)

    def __str__(self):
        return np.array2string(self.as_ndarray)

    def __repr__(self):
        return self.as_tuples

    @cached_property
    def factors(self):
        return factorize_monomial(self.as_ndarray)

    @cached_property
    def knowable_factors(self):
        return [tuple(tuple(vec) for vec in np.take(factor, [0, -1, 2], axis=1))
                for factor in self.factors if self.knowable_q(factor)]

    @cached_property
    def unknowable_factors(self):
        raw_unknowns = tuple(filter(self.unknowable_q, self.factors))
        if len(raw_unknowns) < 2:
            if len(raw_unknowns) == 1:
                return self.to_representative(raw_unknowns[0])
            else:
                return tuple()
        else:
            candidate_unknown_part_representatives = []
            for perm in itertools.permutations(raw_unknowns, len(raw_unknowns)):
                candidate_unknown_part_representatives.append(self.to_representative(np.vstack(perm)))
            return sorted(candidate_unknown_part_representatives)[0]

    @cached_property
    def knowability_status(self):
        if len(self.knowable_factors)==len(self.factors):
            return 'Yes'
        elif len(self.unknowable_factors)>0:
            return 'Semi'
        else:
            return 'No'

    @cached_property
    def physical_q(self):
        return all(self.is_physical(factor) for factor in self.factors)



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
