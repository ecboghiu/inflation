"""
This file contains classes for defining the monomials inside a moment matrix.

@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""
import numpy as np

from functools import total_ordering, cached_property
from typing import Tuple

from ..lp.monomial_classes import InternalAtomicMonomial as InternalAtomicMonomialLP
from ..lp.monomial_classes import CompoundMoment as CompoundMomentLP


@total_ordering
class InternalAtomicMonomialSDP(InternalAtomicMonomialLP):
    def __init__(self, inflation_sdp_instance, lexmon: np.ndarray):
        r"""This class models a moment
        :math:`\langle Op_1 Op_2\dots Op_n\rangle` on the inflated problem,
        which cannot be decomposed into products of other moments. It is used
        as a building block for the ``CompoundMonomial`` class. It is
        initialized with a 2D array representing a moment and an an instance of
        ``InflationSDP``, used for methods that depend on the scenario.

        2D Array encoding
        -----------------
        A moment :math:`M=\langle Op_1 Op_2\dots Op_n\rangle` can be specified
        by a 2D array with `n` rows, one for each operator :math:`Op_k`.
        Row `k` contains a list of integers which encode information about the
        operator :math:`Opk`.
         * The first integer is an index in ``{1,...,nr_parties}``, indicating
           the party, where `nr_parties` is the number of parties in the DAG.
         * The second-to-last and last integers encode the setting and the
           outcome of the operator, respectively.
         * The remaining positions ``i`` indicate on which copy of the source
           ``i-1`` (-1 because the first index encodes the party) the operator
           is acting, with value ``0`` representing no support on the
           ``i-1``-th source.

        For example, the moment
        :math:`\langle A^{0,2,1}_{x=2,a=3} C^{2,0,1}_{z=4,c=5}\rangle`, where
        the complete list of parties is ``["A","B","C"]`` corresponds to the
        following array:

        >>> m = np.array([[1, 0, 2, 1, 2, 3],
                          [3, 2, 0, 1, 4, 5]])

        Given that this moment is knowable and can be associated with a
        probability, it is given the name ``"pAC(35|24)"``.

        Parameters
        ----------
        inflation_sdp_instance : InflationSDP
            An instance of the ``InflationSDP`` class. It is used to access
            methods specific to the inflation problem. E.g., when instantiating
            an internal atomic moment, the ``InflationSDP`` instance is used to
            check if it already contains such moment.
        lexmon : numpy.ndarray
            A moment :math:`\langle Op_1Op_2\dots Op_n\rangle` encoded as a 1D
            array lexmon.
        """
        super().__init__(inflation_sdp_instance, lexmon)

    def __copy__(self):
        """Make a copy of the Monomial"""
        cls = self.__class__
        result = cls.__new__(cls)
        for attr in self.__slots__:
            try:
                result.__setattr__(attr, self.__getattribute__(attr))
            except AttributeError:
                pass
        return result

    @cached_property
    def is_all_commuting(self) -> bool:
        """If the moment containts operators that all commute.
        """
        return self.context.all_commuting_q_1d(self.as_lexmon)

    @cached_property
    def conjugate_lexmon(self):
        if self.is_all_commuting:
            return self.as_lexmon
        else:
            return self.context._conjugate_lexmon(self.as_lexmon)

    @cached_property
    def is_hermitian(self) -> bool:
        """Whether the atomic monomial is equivalent to its conjugate
         under inflation symmetries and commutation.
        """
        return np.array_equal(self.as_lexmon, self.conjugate_lexmon)

    @property
    def dagger(self):
        """Returns the adjoint of the Monomial."""
        if self.is_hermitian:
            return self
        else:
            dagger = self.__copy__()
            dagger.as_lexmon = self.conjugate_lexmon
            dagger.signature = (self.n_operators, tuple(self.conjugate_lexmon))
            dagger.as_2d_array = self.context._lexorder[self.conjugate_lexmon]
            dagger.name = "<" + " ".join(
                self.context._lexrepr_to_names[dagger.as_lexmon]) + ">"
            return dagger
        
class CompoundMomentSDP(CompoundMomentLP):
    def __init__(self, monomials: Tuple[InternalAtomicMonomialSDP]):
        r"""This class models moments :math:`\langle Op_1 Op_2\dots Op_n\rangle
        =\langle Op_i\dots\rangle\langle Op_{i'}\dots\rangle` on the inflated
        problem that are products of other moments. It is built from a tuple of
        instances of the ``InternalAtomicMonomial`` class.

        At initialisation, a moment is classified into knowable, semi-knowable
        or unknowable based on the knowability of each of the atomic moments
        (which in turn is determined through methods of the
        ``InternalAtomicMonomial`` class). This class also computes names for
        the moment, provides the ability to compare (in)equivalence, and to
        assign numerical values to a moment given a probability distribution.

        Parameters
        ----------
        monomials : tuple of InternalAtomicMonomialSDP
            The atomic moments that make up the compound moment.
        """
        default_factors    = tuple(sorted(monomials))
        conjugate_factors  = tuple(sorted(factor.dagger
                                          for factor in monomials))
        super().__init__(min(default_factors, conjugate_factors))
        self.internal_type = InternalAtomicMonomialSDP

    @property
    def is_all_commuting(self):
        """If all factors of the compount moment contain operators that all
        commute."""
        return all(factor.is_all_commuting for factor in self.factors)

    @property
    def is_physical(self):
        """If all factors of the compount moment contain monomials that are 
        physical, i.e., products of positive operators that are positive."""
        return all(factor.is_physical for factor in self.factors)

    @cached_property
    def is_hermitian(self):
        """If all factors of the compount moment contain monomials that are 
        hermitian."""
        return all(factor.is_hermitian for factor in self.factors)