"""
This file contains classes for defining the monomials inside a moment matrix.

@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""
import numpy as np

from collections import Counter
from functools import total_ordering
from numbers import Real
from typing import Dict, List, Tuple

from .fast_npa import mon_is_zero, nb_remove_sandwich
from .monomial_utils import (compute_marginal,
                             name_from_atom_names,
                             symbol_from_atom_name,
                             symbol_prod)


@total_ordering
class InternalAtomicMonomial(object):
    __slots__ = ["as_ndarray",
                 "is_one",
                 "is_zero",
                 "is_knowable",
                 "is_all_commuting",
                 "is_physical",
                 "n_operators",
                 "op_length",
                 "rectified_ndarray",
                 "sdp"
                 ]

    def __init__(self, inflation_sdp_instance, array2d: np.ndarray):
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
        array2d : numpy.ndarray
            A moment :math:`\langle Op_1Op_2\dots Op_n\rangle` encoded as a 2D
            array.
        """
        self.sdp        = inflation_sdp_instance
        self.as_ndarray = np.asarray(array2d, dtype=self.sdp.np_dtype)
        self.n_operators, self.op_length = self.as_ndarray.shape
        assert self.op_length == self.sdp._nr_properties, \
            ("An AtomicMonomial should be a 2-d array where each row is a list"
             + f" of integers of length {self.sdp._nr_properties}. The first "
             + "index corresponds to the party, the last one to the outcome, "
             + "the second-to-last to the setting, and the rest to the "
             + "inflation copies.")
        self.is_zero     = mon_is_zero(self.as_ndarray)
        self.is_one      = (self.n_operators == 0)
        self.is_knowable = (self.is_zero
                            or self.is_one
                            or self.sdp._atomic_knowable_q(self.as_ndarray))
        self.is_all_commuting = self.sdp.all_commuting_q(self.as_ndarray)
        if self.is_all_commuting or self.n_operators <= 1:
            self.is_physical = True
        else:
            self.is_physical = self.sdp.all_commuting_q(
                nb_remove_sandwich(self.as_ndarray))
        # Save also array with the original setting, not just the effective one
        if self.is_knowable:
            self.rectified_ndarray = np.asarray(
                self.sdp.rectify_fake_setting(np.take(self.as_ndarray,
                                                      [0, -2, -1],
                                                      axis=1)),
                dtype=int)

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
        conjugate_ndarray   = self.sdp._conjugate_ndarray(self.as_ndarray)
        conjugate_signature = self.sdp._from_2dndarray(conjugate_ndarray)
        if conjugate_signature != self.signature:
            dagger = self.__copy__()
            dagger.as_ndarray = conjugate_ndarray
            return dagger
        else:
            return self

    @property
    def is_hermitian(self):
        """Whether the atomic monomial is equivalent to its conjugate
         under inflation symmetries and commutation.
        """
        return self == self.dagger

    @property
    def name(self):
        """A string representing the monomial. In case of knowable monomials,
        it is of the form ``p(outputs|inputs)``. Otherwise it represents the
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
            p_divider = ""
            # We will probably never have more than 1 digit cardinalities,
            # but who knows...
            i_divider = "" if all(len(i) == 1 for i in inputs) else ","
            o_divider = "" if all(len(o) == 1 for o in outputs) else ","
            return ("p" + p_divider.join(parties) +
                    "(" + o_divider.join(outputs) +
                    "|" + i_divider.join(inputs) + ")")
        else:
            # Use expectation value notation
            operators = []
            for op in self.as_ndarray:
                operators.append("_".join([self.sdp.names[op[0] - 1]]
                                          + [str(i) for i in op[1:]]))
            return "<" + " ".join(operators) + ">"

    @property
    def signature(self):
        return self.sdp._from_2dndarray(self.as_ndarray)

    @property
    def symbol(self):
        """Return a sympy Symbol representing the monomial."""
        return symbol_from_atom_name(self.name)

    def compute_marginal(self, prob_array: np.ndarray) -> float:
        """Given a probability distribution, compute the numerical value of
        the Monomial.

        Parameters
        ----------
        prob_array : numpy.ndarray
            The target probability distribution. The dimensions of this array
            are (outcomes_per_party, settings_per_party), where the settings
            are explicitly 1 is there is only one measurement performed.

        Returns
        -------
        float
            The value of the corresponding probability (which can be a marginal
            involving only a few parties)
        """
        if self.is_zero:
            return 0.
        else:
            return compute_marginal(prob_array, self.rectified_ndarray)


class CompoundMonomial(object):
    __slots__ = ["as_counter",
                 "factors",
                 "idx",
                 "is_all_commuting",
                 "is_atomic",
                 "is_knowable",
                 "is_physical",
                 "is_one",
                 "is_zero",
                 "knowability_status",
                 "knowable_factors",
                 "mask_matrix",
                 "n_factors",
                 "n_knowable_factors",
                 "n_operators",
                 "n_unknowable_factors",
                 "name",
                 "signature",
                 "symbol",
                 "unknowable_factors"
                 ]

    def __init__(self, monomials: Tuple[InternalAtomicMonomial]):
        r"""This class models moments :math:`\langle Op_1 Op_2\dots Op_n\rangle
        =\langle Op_i\dots\rangle\langle Op_{i'}\dots\rangle` on the inflated
        problem that are products of other moments. It is built from a tuple of
        instances of the ``InternalAtomicMonomial`` class.

        At intialisation, a moment is classified into knowable, semi-knowable
        or unknowable based on the knowability of each of the atomic moments
        (which in turn is determined through methods of the
        ``InternalAtomicMonomial`` class). This class also computes names for
        the moment, provides the ability to compare (in)equivalence, and to
        assign numerical values to a moment given a probability distribution.

        Parameters
        ----------
        monomials : tuple of InternalAtomicMonomial
            The atomic moments that make up the compound moment.
        """
        default_factors    = tuple(sorted(monomials))
        conjugate_factors  = tuple(sorted(factor.dagger
                                          for factor in monomials))
        self.factors       = min(default_factors, conjugate_factors)
        self.n_factors     = len(self.factors)
        self.n_operators   = sum(factor.n_operators for factor in self.factors)
        self.is_atomic     = (self.n_factors <= 1)
        self.is_knowable   = all(factor.is_knowable for factor in self.factors)
        self.is_all_commuting = all(factor.is_all_commuting
                                    for factor in self.factors)
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
            self.knowability_status = "Knowable"
        elif self.n_unknowable_factors == self.n_factors:
            self.knowability_status = "Unknowable"
        else:
            self.knowability_status = "Semi"
        self.is_zero = any(factor.is_zero for factor in self.factors)
        self.is_one  = (all(factor.is_one for factor in self.factors)
                        or (self.n_factors == 0))
        self.is_physical = all(factor.is_physical for factor in self.factors)
        self.as_counter  = Counter(self.factors)
        self.name        = name_from_atom_names(self._names_of_factors)
        self.symbol      = symbol_prod(self._symbols_of_factors)
        self.signature   = tuple(sorted(self.factors))

    def __eq__(self, other):
        """Whether the Monomial is equal to the ``other`` Monomial."""
        if isinstance(other, self.__class__):
            return self.as_counter == other.as_counter
        elif isinstance(other, InternalAtomicMonomial):
            return (self.n_factors == 1) and other.__eq__(self.factors[0])
        else:
            assert isinstance(other, self.__class__), \
                (f"Expected object of class {self.__class__}, received {other}"
                 + f" of class {type(other)}{list(map(type, other))}.")
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
    def _names_of_factors(self):
        """Return the names of each of the factors in the Monomial."""
        return [factor.name for factor in self.factors]

    @property
    def _symbols_of_factors(self):
        """Generate a sympy Symbol per factor in the Monomial."""
        return [factor.symbol for factor in self.factors]

    def attach_idx(self, idx: int):
        """Assign an index to the Monomial. This is used when generating the
        monomials in a scenario, and identifying them with integers."""
        if idx >= 0:
            self.idx = idx

    def compute_marginal(self, prob_array: np.ndarray) -> float:
        """Given a probability array, evaluate all the knowable atomic moment
        factors in the monomial and return the product of the resulting values.

        The whole moment needs to be knowable (``self.is_knowable = True``),
        else this method will raise an error.

        Parameters
        ----------
        prob_array : np.ndarray
            A conditional probability distribution over the non-inflated
            scenario with dimensions ``(da,db,...,dx,dy,...)`` (i.e., first
            outcomes and then settings for the parties).

        See also
        --------
        CompoundMonomial.evaluate : This function does not rely on the
            knowability of the monomial.
        """
        assert self.is_knowable, ("Only marginals of knowable monomials can " +
                                  f"be computed, and {self} is not knowable.")
        value = 1.
        for factor, power in self.as_counter.items():
            value *= (factor.compute_marginal(prob_array) ** power)
        return value

    def evaluate(self,
                 known_monomials: Dict[InternalAtomicMonomial, float],
                 use_lpi_constraints=True) -> Tuple[float, List, str]:
        """Given a dictionary of values for known atomic monomials,
        substitute all factors of the compound moment that are specified in
        the dictionary with their values and return the product of the values
        and a remainder compound moment made of the remaining factors.

        Parameters
        ----------
        known_monomials : Dict[InternalAtomicMonomial, float]
            A dictionary of known atomic monomials and their values.
        use_lpi_constraints : bool, optional
            Whether compound moments whose factors are partially present in
            ``known_monomials`` are labelled as ``"Semi"`` or ``"Unknowable"``.
            If ``True``, they are labelled as ``"Semi"``. By default ``True``.

        Returns
        -------
        Tuple[float, List, str]
            A tuple where the first element is the value of the knowable parts
            of the compound moment, the second element is a compound monomial
            of all the remaining factors not present in ``known_monomials`` and
            the third element is a string describing whether all factors are
            present in ``known_monomials`` (``"Known"``), some are
            (``"Semi"``), or none are (``"Unknown"``).
        """
        known_value     = 1.
        unknown_counter = Counter()
        for factor, power in self.as_counter.items():
            try:
                known_value *= (known_monomials[factor] ** power)
            except KeyError:
                unknown_counter[factor] = power
        unknown_factors = list(unknown_counter.elements())
        if (len(unknown_factors) == 0):
            known_status = "Known"
        elif ((len(unknown_factors) == self.n_factors)
              or (not use_lpi_constraints)):
            known_status = "Unknown"
        else:
            known_status = "Semi"
            if use_lpi_constraints and isinstance(known_value, Real):
                if np.isclose(known_value, 0):
                    known_status = "Known"
        return known_value, unknown_factors, known_status
