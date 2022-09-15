"""
The module generates the semidefinite program associated to a quantum inflation
instance (see arXiv:1909.10519).

@authors: Alejandro Pozas-Kerstjens, Emanuel-Cristian Boghiu
"""
import gc
import itertools
import numbers
import warnings
from collections import Counter

import numpy as np
import sympy as sp

from causalinflation import InflationProblem
from typing import List, Dict, Tuple, Union, Any
from .fast_npa import (calculate_momentmatrix,
                       to_canonical,
                       to_name,
                       remove_projector_squares,
                       mon_lexsorted,
                       mon_is_zero,
                       nb_mon_to_lexrepr,
                       notcomm_from_lexorder)
from .general_tools import (to_representative,
                            to_numbers,
                            to_symbol,
                            flatten,
                            flatten_symbolic_powers,
                            phys_mon_1_party_of_given_len,
                            is_knowable,
                            find_permutation,
                            apply_source_permutation_coord_input,
                            generate_operators,
                            clean_coefficients,
                            factorize_monomial
                            )
from .monomial_classes import InternalAtomicMonomial, CompoundMonomial
from .sdp_utils import solveSDP_MosekFUSION
from .writer_utils import (write_to_csv, write_to_mat, write_to_sdpa)

# Force warnings.warn() to omit the source code line in the message
# Source: https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
    formatwarning_orig(message, category, filename, lineno, line='')


from scipy.sparse import coo_matrix

try:
    from numba import types
    from numba.typed import Dict as nb_Dict
    from tqdm import tqdm
except ImportError:
    from causalinflation.utils import blank_tqdm as tqdm


class InflationSDP(object):
    """
    Class for generating and solving an SDP relaxation for quantum inflation.

    Parameters
    ----------
    inflationproblem : InflationProblem
        Details of the scenario.
    commuting : bool, optional
        Whether variables in the problem are going to be commuting (classical
        problem) or non-commuting (quantum problem). By default ``False``.
    verbose : int, optional
        Optional parameter for level of verbose:

            * 0: quiet (default),
            * 1: monitor level: track program process,
            * 2: debug level: show properties of objects created.
    """

    def __init__(self, inflationproblem: InflationProblem,
                 commuting: bool = False,
                 supports_problem: bool = False,
                 verbose: int = 0):
        """Constructor for the InflationSDP class.
        """
        self.supports_problem = supports_problem
        self.verbose = verbose
        self.commuting = commuting
        self.InflationProblem = inflationproblem
        self.names = self.InflationProblem.names
        if self.verbose > 1:
            print(self.InflationProblem)

        self.nr_parties = len(self.names)
        self.nr_sources = self.InflationProblem.nr_sources
        self.hypergraph = self.InflationProblem.hypergraph
        self.inflation_levels = self.InflationProblem.inflation_level_per_source
        if self.supports_problem:
            self.outcome_cardinalities = self.InflationProblem.outcomes_per_party + 1
        else:
            self.outcome_cardinalities = self.InflationProblem.outcomes_per_party
        self.setting_cardinalities = self.InflationProblem.settings_per_party

        self._generate_parties()

        self.maximize = True  # Direction of the optimization
        self.split_node_model = self.InflationProblem.split_node_model
        self.is_knowable_q_split_node_check = self.InflationProblem.is_knowable_q_split_node_check
        self.rectify_fake_setting_atomic_factor = self.InflationProblem.rectify_fake_setting_atomic_factor

        self._nr_operators = len(flatten(self.measurements))
        self._nr_properties = 1 + self.nr_sources + 2
        self.np_dtype = np.find_common_type([
            np.min_scalar_type(np.max(self.setting_cardinalities)),
            np.min_scalar_type(np.max(self.outcome_cardinalities)),
            np.min_scalar_type(self.nr_parties+1),
            np.min_scalar_type(np.max(self.inflation_levels)+1)], [])
        self.identity_operator = np.empty((0, self._nr_properties), dtype=self.np_dtype)
        self.zero_operator = np.zeros((1, self._nr_properties), dtype=self.np_dtype)

        # Define default lexicographic order through np.lexsort
        # The lexicographic order is encoded as a matrix with rows as
        # operators and the row index gives the order
        arr = np.array([to_numbers(op, self.names)[0]
                        for op in flatten(self.measurements)], dtype=self.np_dtype)
        self._default_lexorder = arr[np.lexsort(np.rot90(arr))]
        self._lexorder = self._default_lexorder.copy()

        # Given that most operators commute, we want the matrix encoding the
        # commutations to be sparse, so self._default_commgraph[i, j] = 0
        # implies commutation, and self._default_commgraph[i, j] = 1 is
        # non-commutation.
        self._default_notcomm = notcomm_from_lexorder(self._lexorder)
        self._notcomm = self._default_notcomm.copy()

        self.canonsym_ndarray_from_hash_cache = dict()
        self.atomic_monomial_from_hash_cache = dict()
        self.compound_monomial_from_tuple_of_atoms_cache = dict()
        self.compound_monomial_from_name_dict = dict()
        self.Zero = self.Monomial(self.zero_operator, idx=0)
        self.One = self.Monomial(self.identity_operator, idx=1)

    def AtomicMonomial(self, array2d: np.ndarray) -> InternalAtomicMonomial:
        quick_key = self.from_2dndarray(array2d)
        try:
            return self.atomic_monomial_from_hash_cache[quick_key]
        except KeyError:  # Key not in atomic_monomial cache
            try:
                new_array2d = self.canonsym_ndarray_from_hash_cache[quick_key]
                new_quick_key = self.from_2dndarray(new_array2d)
                try:  # Key is in universal cache
                    new_mon = self.atomic_monomial_from_hash_cache[new_quick_key]
                    self.atomic_monomial_from_hash_cache[quick_key] = new_mon
                    return new_mon
                except KeyError:  # Key is in universal cache, but not yet in the atomic cache
                    new_mon = InternalAtomicMonomial(inflation_sdp_instance=self, array2d=new_array2d)
                    self.atomic_monomial_from_hash_cache[quick_key] = new_mon
                    self.atomic_monomial_from_hash_cache[new_quick_key] = new_mon
                    return new_mon
            except KeyError:  # Key not in atomic_monomial cache NOR in universal_cache
                if len(array2d) == 0 or np.array_equiv(array2d, 0):
                    new_array2d = array2d
                else:
                    new_array2d = self.inflation_aware_to_ndarray_representative(array2d)
                new_quick_key = self.from_2dndarray(new_array2d)
                new_mon = InternalAtomicMonomial(inflation_sdp_instance=self, array2d=new_array2d)
                self.canonsym_ndarray_from_hash_cache[quick_key] = new_array2d
                self.canonsym_ndarray_from_hash_cache[new_quick_key] = new_array2d
                self.atomic_monomial_from_hash_cache[quick_key] = new_mon
                self.atomic_monomial_from_hash_cache[new_quick_key] = new_mon
                return new_mon

    def monomial_from_list_of_atomic(self, list_of_AtomicMonomials: List[InternalAtomicMonomial]):
        list_of_atoms = []
        for factor in list_of_AtomicMonomials:
            if factor.is_zero:
                list_of_atoms = [factor]
                break
            elif not factor.is_one:
                list_of_atoms.append(factor)
            else:
                pass
        tuple_of_atoms = tuple(sorted(list_of_atoms))
        try:
            mon = self.compound_monomial_from_tuple_of_atoms_cache[tuple_of_atoms]
        except KeyError:
            mon = CompoundMonomial(tuple_of_atoms)
            self.compound_monomial_from_tuple_of_atoms_cache[tuple_of_atoms] = mon
            self.compound_monomial_from_name_dict[mon.name] = mon
        return mon

    def Monomial(self, array2d: np.ndarray, idx=-1) -> CompoundMonomial:
        _factors = factorize_monomial(array2d, canonical_order=False)
        list_of_atoms = [self.AtomicMonomial(factor) for factor in _factors if len(factor)]
        mon = self.monomial_from_list_of_atomic(list_of_atoms)
        mon.attach_idx_to_mon(idx)
        return mon

    def inflation_aware_knowable_q(self, atomic_monarray: np.ndarray) -> bool:
        if self.split_node_model:
            minimal_monomial = tuple(tuple(vec) for vec in np.take(atomic_monarray, [0, -2, -1], axis=1))
            return self.is_knowable_q_split_node_check(minimal_monomial)
        else:
            return True

    def atomic_knowable_q(self, atomic_monarray: np.ndarray) -> bool:
        first_test = is_knowable(atomic_monarray)
        if not first_test:
            return False
        else:
            return self.inflation_aware_knowable_q(atomic_monarray)

    def inflation_aware_to_ndarray_representative(self, mon: np.ndarray,
                                                  swaps_plus_commutations=True,
                                                  consider_conjugation_symmetries=True) -> np.ndarray:
        unsym_monarray = to_canonical(mon, self._notcomm, self._lexorder)
        quick_key = self.from_2dndarray(unsym_monarray)
        try:
            sym_monarray = self.canonsym_ndarray_from_hash_cache[quick_key]
            if self.verbose > 0:
                warnings.warn("This 'to_representative' function should only be called as a last resort.")
        except KeyError:
            sym_monarray = to_representative(unsym_monarray,
                                             self.inflation_levels,
                                             self._notcomm,
                                             self._lexorder,
                                             swaps_plus_commutations=swaps_plus_commutations,
                                             consider_conjugation_symmetries=consider_conjugation_symmetries,
                                             commuting=self.commuting)
            new_quick_key = self.from_2dndarray(sym_monarray)
            if new_quick_key not in self.canonsym_ndarray_from_hash_cache:
                if self.verbose > 0:
                    warnings.warn(
                        f"Encountered a monomial that does not appear in the original moment matrix:\n {sym_monarray}")
            self.canonsym_ndarray_from_hash_cache[new_quick_key] = sym_monarray
        self.canonsym_ndarray_from_hash_cache[quick_key] = sym_monarray
        return sym_monarray

    def from_2dndarray(self, array2d: np.ndarray):
        return np.asarray(array2d, dtype=self.np_dtype).tobytes()

    def to_2dndarray(self, bytestream):
        return np.frombuffer(bytestream, dtype=self.np_dtype).reshape((-1, self._nr_properties))

    def commutation_relationships(self):
        """This returns a user-friendly representation of the commutation relationships."""
        from collections import namedtuple
        nonzero = namedtuple('NonZeroExpressions', 'exprs')
        data = []
        for i in range(self._lexorder.shape[0]):
            for j in range(i, self._lexorder.shape[0]):
                # Most operators commute as they belong to different parties,
                # so it is more interested to list those that DON'T commute.
                if self._notcomm[i, j] != 0:
                    op1 = sp.Symbol(to_name([self._lexorder[i]], self.names), commutative=False)
                    op2 = sp.Symbol(to_name([self._lexorder[i]], self.names), commutative=False)
                    if self.verbose > 0:
                        print(f"{str(op1 * op2 - op2 * op1)} ≠ 0.")
                    data.append(op1 * op2 - op2 * op1)
        return nonzero(data)

    def lexicographic_order(self) -> dict:
        """This returns a user-friendly representation of the lexicographic order."""
        lexicographic_order = {}
        for i, op in enumerate(self._lexorder):
            lexicographic_order[sp.Symbol(to_name([op], self.names),
                                          commutative=False)] = i
        return lexicographic_order


    ########################################################################
    # MAIN ROUTINES EXPOSED TO THE USER                                    #
    ########################################################################
    def generate_relaxation(self,
                            column_specification:
                            Union[str,
                                  List[List[int]],
                                  List[sp.core.symbol.Symbol]] = 'npa1'
                            ) -> None:
        r"""Creates the SDP relaxation of the quantum inflation problem using
        the `NPA hierarchy <https://www.arxiv.org/abs/quant-ph/0607119>`_ and
        applies the symmetries inferred from inflation.

        It takes as input the generating set of monomials :math:`\{M_i\}_i`. The
        moment matrix :math:`\Gamma` is defined by all the possible inner
        products between these monomials:

        .. math::

            \Gamma[i, j] := \operatorname{tr} (\rho \cdot M_i^\dagger M_j).

        The set :math:`\{M_i\}_i` is specified by the parameter
        ``column_specification``.

        In the inflated graph there are many symmetries coming from invariance
        under swaps of the copied sources, which are used to remove variables
        in the moment matrix.

        Parameters
        ----------
        column_specification : Union[str, List[List[int]], List[sympy.core.symbol.Symbol]]
            Describes the generating set of monomials :math:`\{M_i\}_i`.

            * `(str)` ``'npaN'``: where N is an integer. This represents level N
              in the Navascues-Pironio-Acin hierarchy (`arXiv:quant-ph/0607119
              <https://www.arxiv.org/abs/quant-ph/0607119>`_).
              For example, level 3 with measurements :math:`\{A, B\}` will give
              the set :math:`{1, A, B, AA, AB, BB, AAA, AAB, ABB, BBB\}` for
              all inflation, input and output indices. This hierarchy is known
              to converge to the quantum set for :math:`N\rightarrow\infty`.

            * `(str)` ``'localN'``: where N is an integer. Local level N
              considers monomials that have at most N measurement operators per
              party. For example, ``local1`` is a subset of ``npa2``; for two
              parties, ``npa2`` is :math:`\{1, A, B, AA, AB, BB\}` while
              ``local1`` is :math:`\{1, A, B, AB\}`.

            * `(str)` ``'physicalN'``: The subset of local level N with only
              operators that have non-negative expectation values with any
              state. N cannot be greater than the smallest number of copies of a
              source in the inflated graph. For example, in the scenario
              A-source-B-source-C with 2 outputs and no inputs, ``physical2``
              only gives 5 possibilities for B: :math:`\{1, B^{1,1}_{0|0},
              B^{2,2}_{0|0}, B^{1,1}_{0|0}B^{2,2}_{0|0},
              B^{1,2}_{0|0}B^{2,1}_{0|0}\}`. There are no other products where
              all operators commute. The full set of physical generating
              monomials is built by taking the cartesian product between all
              possible physical monomials of each party.

            * `List[List[int]]`: This encodes a party block structure.
              Each integer encodes a party. Within a party block, all missing
              input, output and inflation indices are taken into account. For
              example, ``[[], [0], [1], [0, 1]]`` gives the set :math:`\{1, A,
              B, AB\}`, which is the same as ``local1``. The set ``[[], [0],
              [1], [2], [0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]`` is the
              same as :math:`\{1, A, B, C, AA, AB, AC, BB, BC, CC\}`, which is
              the same as ``npa2`` for three parties. ``[[]]`` encodes the
              identity element.

            * `List[sympy.core.symbol.Symbol]`: one can also fully specify the
              generating set by giving a list of symbolic operators built from
              the measurement operators in `self.measurements`. This list needs
              to have the identity ``sympy.S.One`` as the first element.
        """
        # Process the column_specification input and store the result
        # in self.generating_monomials.
        self.generating_monomials_sym, self.generating_monomials = \
            self.build_columns(column_specification,
                               return_columns_numerical=True)

        if self.verbose > 1:
            print("Number of columns:", len(self.generating_monomials))

        # Calculate the moment matrix without the inflation symmetries.
        self.unsymmetrized_mm_idxs, self.unsymidx_to_unsym_monarray_dict = self._build_momentmatrix()
        if self.verbose > 1:
            print("Number of variables before symmetrization:",
                  len(self.unsymidx_to_unsym_monarray_dict))

        _unsymidx_from_hash_dict = {self.from_2dndarray(v): k for (k, v) in
                                    self.unsymidx_to_unsym_monarray_dict.items()}

        # Calculate the inflation symmetries.
        self.inflation_symmetries = self._calculate_inflation_symmetries()

        # Apply the inflation symmetries to the moment matrix.
        self.momentmatrix, self.orbits, self.symidx_to_sym_monarray_dict \
            = self._apply_inflation_symmetries(self.unsymmetrized_mm_idxs,
                                               self.unsymidx_to_unsym_monarray_dict,
                                               self.inflation_symmetries)
        for (k, v) in _unsymidx_from_hash_dict.items():
            self.canonsym_ndarray_from_hash_cache[k] = self.symidx_to_sym_monarray_dict[self.orbits[v]]
        del _unsymidx_from_hash_dict

        self.largest_moment_index = max(self.symidx_to_sym_monarray_dict.keys())

        self.list_of_monomials = []
        (self.momentmatrix_has_a_zero, self.momentmatrix_has_a_one) = np.in1d([0, 1], self.momentmatrix.ravel())
        if self.momentmatrix_has_a_zero:
            self.list_of_monomials.append(self.Zero)
        #The zero monomial is not stored during calculate_momentmatrix, so we manually add it here.
        self.list_of_monomials.extend([self.Monomial(v, idx=k)
                                       for (k, v) in self.symidx_to_sym_monarray_dict.items() if k>0])
        for mon in self.list_of_monomials:
            mon.mask_matrix = coo_matrix(self.momentmatrix == mon.idx).tocsr()
        """
        Used only for internal diagnostics.
        """
        _counter = Counter([mon.knowability_status for mon in self.list_of_monomials])
        self.n_knowable = _counter['Yes']
        self.n_something_knowable = _counter['Semi']
        self.n_unknowable = _counter['No']

        if self.commuting:
            self.possibly_physical_monomials = self.list_of_monomials
        else:
            self.possibly_physical_monomials = [mon for mon in self.list_of_monomials if mon.physical_q]

        # This is useful for the certificates
        self.name_dict_of_monomials = {mon.name: mon for mon in self.list_of_monomials}
        self.monomial_names = list(self.name_dict_of_monomials.keys())

        self.maskmatrices_name_dict = {mon.name: mon.mask_matrix for mon in self.list_of_monomials}
        self.maskmatrices = {mon: mon.mask_matrix for mon in self.list_of_monomials}

        self.moment_linear_equalities = []
        self.moment_linear_inequalities = []
        self.moment_upperbounds = dict()
        self.moment_lowerbounds = {m: 0. for m in self.possibly_physical_monomials}

        self.set_objective(None)  # Equivalent to reset_objective
        self.set_values(None)  # Equivalent to reset_values


    def reset_objective(self):
        for attribute in {'objective', '_objective_as_name_dict', 'objective_value', '_processed_objective'}:
            try:
                delattr(self, attribute)
            except AttributeError:
                pass
        self.objective = {self.One: 0.}
        self._objective_as_name_dict = {'1': 0.}

    def reset_values(self):
        for attribute in {'known_moments', 'semiknown_moments', '_processed_moment_lowerbounds',
                          'known_moments_name_dict', 'semiknown_moments_name_dict',
                          '_processed_moment_lowerbounds_name_dict'}:
            try:
                delattr(self, attribute)
            except AttributeError:
                pass
        gc.collect(2)
        self.known_moments = dict()
        self.semiknown_moments = dict()
        self._processed_moment_lowerbounds = dict()
        self.known_moments_name_dict = dict()
        self.semiknown_moments_name_dict = dict()
        self._processed_moment_lowerbounds_name_dict = dict()
        if self.momentmatrix_has_a_zero:
            self.known_moments[self.Zero] = 0.
            self.known_moments_name_dict[self.Zero.name] = 0.

    def update_physical_lowerbounds(self):
        for mon in set(self.moment_lowerbounds.keys()).difference(self.known_moments.keys()):
            self._processed_moment_lowerbounds[mon] = self.moment_lowerbounds[mon]
            self._processed_moment_lowerbounds_name_dict = {mon.name: value for mon, value in
                                                            self._processed_moment_lowerbounds.items()}

    def set_distribution(self,
                         prob_array: Union[np.ndarray, None],
                         use_lpi_constraints: bool = False,
                         assume_shared_randomness: bool = False) -> None:
        """Set numerically all the knowable (and optionally semiknowable)
        moments according to the probability distribution
        specified.

        Parameters
        ----------
            prob_array : numpy.ndarray
                Multidimensional array encoding the distribution, which is
                called as ``prob_array[a,b,c,...,x,y,z,...]`` where
                :math:`a,b,c,\dots` are outputs and :math:`x,y,z,\dots` are
                inputs. Note: even if the inputs have cardinality 1 they must be
                specified, and the corresponding axis dimensions are 1.

            use_lpi_constraints : bool, optional
                Specification whether linearized polynomial constraints (see,
                e.g., Eq. (D6) in `arXiv:2203.16543
                <http://www.arxiv.org/abs/2203.16543/>`_) will be imposed or not.
                By default ``False``.

            assume_shared_randomness (bool): Specification whether higher order monomials
                may be calculated. If universal shared randomness is present, only atomic
                monomials may be evaluated from the distribution.
        """
        knowable_values = {m: m.compute_marginal(prob_array) for m in self.list_of_monomials
                           if m.is_atomic and m.knowable_q} if (prob_array is not None) else dict()
        # Compute self.known_moments and self.semiknown_moments and names their corresponding names dictionaries
        self.set_values(knowable_values, use_lpi_constraints=use_lpi_constraints,
                        only_knowable_moments=(not use_lpi_constraints),
                        only_specified_values=assume_shared_randomness,
                        consider_only_semiknowable=True)


    def set_values(self, values: Union[
        Dict[Union[sp.core.symbol.Symbol, str, CompoundMonomial, InternalAtomicMonomial], float], None],
                   use_lpi_constraints: bool = False,
                   normalised: bool = True,
                   only_knowable_moments: bool = True,
                   only_specified_values: bool = False,
                   consider_only_semiknowable: bool = False,
                   ) -> None:
        """Directly assign numerical values to variables in the moment matrix.
        This is done via a dictionary where keys are the variables to have
        numerical values assigned (either in their operator form, in string
        form, or directly referring to the variable in the moment matrix), and
        the values are the corresponding numerical quantities.

        Parameters
        ----------
        values : Dict[Union[sympy.core.symbol.Symbol, str, Monomial], float]
            The description of the variables to be assigned numerical values and
            the corresponding values. The keys can be either of the Monomial class,
            symbols or strings (which should be the name of some Monomial).

        use_lpi_constraints : bool
            Specification whether linearized polynomial constraints (see, e.g.,
            Eq. (D6) in arXiv:2203.16543) will be imposed or not.

        only_specified_values : bool
            Specifies whether one wishes to fix only the variables provided (True),
            or also the variables containing products of the monomials fixed (False).
            If only_specified_values is True, unknowable variables can also be fixed.

        only_knowable_moments : bool
            Default true. Set false to allow the user to also specify values of
            monomials that are not a priori knowable.

        normalised: bool
            Specifies whether the unit monomial '1' is given value 1.0 even if
            '1' is not included in the values dictionary (default, True), or if
            is left as a free variable (False).
        """

        self.reset_values()
        if normalised and self.momentmatrix_has_a_one:
            self.known_moments[self.One] = 1
        if (values is None) or (len(values) == 0):
            self.cleanup_after_set_values()
            return

        self.use_lpi_constraints = use_lpi_constraints

        if (len(self.objective) > 1) and self.use_lpi_constraints:
            warnings.warn("You have an objective function set. Be aware that imposing " +
                          "linearized polynomial constraints will constrain the " +
                          "optimization to distributions with fixed marginals.")

        for (k, v) in values.items():
            if not np.isnan(v):
                self.known_moments[self._sanitise_monomial(k)] = v

        # Check that the keys are consistent with the flags set
        if not only_specified_values:
            for k in self.known_moments:
                if not k.is_atomic:
                    raise Exception("set_values: The monomial " + str(k) + " is not an " +
                                    "atomic monomial, but composed of several factors. " +
                                    "Please provide values only for atomic monomials. " +
                                    "If you want to manually be able to set values for " +
                                    "non-atomic monomials, set only_specified_values to True.")
        if only_knowable_moments:
            for k in self.known_moments:
                if not k.knowable_q:
                    raise Exception("set_values: The monomial " + str(k) + " is not an " +
                                    "atomic monomial, but composed of several factors. " +
                                    "Please provide values only for atomic monomials. " +
                                    "If you want to manually be able to set values for " +
                                    "non-atomic monomials, set only_specified_values to True.")

        if only_specified_values:
            # If only_specified_values=True, then ONLY the Monomials that
            # are keys in the values dictionary are fixed. Any other monomial
            # that is semi-known relative to the information in the dictionary
            # is left free.
            if self.use_lpi_constraints and self.verbose >= 1:
                warnings.warn(
                    "set_values: Both only_specified_values=True and use_lpi_constraints=True has been detected. "
                    "With only_specified_values=True, only moments that match exactly " +
                    "those provided in the values dictionary will be set. Values for moments " +
                    "that are products of others moments will not be inferred automatically, " +
                    "and neither will proportionality constraints between moments (LPI constraints). " +
                    "Set only_specified_values=False for these features.")
            self.cleanup_after_set_values()
            return

        atomic_known_moments = {mon.knowable_factors[0]: val for mon, val in self.known_moments.items() if
                                (len(mon) == 1)}
        if only_knowable_moments:
            remaining_monomials_to_compute = (mon for mon in self.list_of_monomials if
                                              (not mon.is_atomic) and mon.knowable_q)  # as iterator, saves memory.
        elif consider_only_semiknowable:
            remaining_monomials_to_compute = (mon for mon in self.list_of_monomials if
                                              (not mon.is_atomic) and mon.knowability_status in ['Yes', 'Semi'])  # as iterator, saves memory.
        else:
            remaining_monomials_to_compute = (mon for mon in self.list_of_monomials if not mon.is_atomic)
        for mon in remaining_monomials_to_compute:
            if mon not in self.known_moments.keys():
                value, unknown_atomic_factors, known_status = mon.evaluate_given_atomic_monomials_dict(
                    atomic_known_moments,
                    use_lpi_constraints=self.use_lpi_constraints)
                if known_status == 'Yes':
                    self.known_moments[mon] = value
                elif known_status == 'Semi':
                    if self.use_lpi_constraints:
                        self.semiknown_moments[mon] = (value, self.monomial_from_list_of_atomic(unknown_atomic_factors))
                else:
                    pass
        del atomic_known_moments
        del remaining_monomials_to_compute
        self.cleanup_after_set_values()
        return

    def cleanup_after_set_values(self):
        # Name dictionaries for compatibility purposes only
        self.known_moments_name_dict = {mon.name: v for mon, v in self.known_moments.items()}
        self.semiknown_moments_name_dict = {mon.name: (value, unknown.name) for mon, (value, unknown) in
                                            self.semiknown_moments.items()}

        if self.supports_problem:
            # Convert positive known values into lower bounds.
            nonzero_known_monomials = [mon for mon, value in self.known_moments.items() if not np.isclose(value, 0)]
            for mon in nonzero_known_monomials:
                self.moment_lowerbounds[mon] = 1.
                del self.known_moments[mon]
            self.semiknown_moments = dict()
            # Name dictionaries for compatibility purposes only, block in code for easy commenting out.
            nonzero_known_monomial_names = [name for name, value in self.known_moments_name_dict.items() if
                                            not np.isclose(value, 0)]
            for name in nonzero_known_monomial_names:
                self.moment_lowerbounds_name_dict[name] = 1.
                del self.known_moments_name_dict[name]
            self.semiknown_moments_name_dict = dict()

        # Create lowerbounds list for physical but unknown moments
        self.update_physical_lowerbounds()
        self._update_objective()
        return

    def set_objective(self,
                      objective: Union[sp.core.symbol.Symbol, None],
                      direction: str = 'max') -> None:
        """Set or change the objective function of the polynomial optimization
        problem.

        Parameters
        ----------
        objective : sympy.core.symbol.Symbol
            Describes the objective function.
        direction : str, optional
            Direction of the optimization (``'max'``/``'min'``). By default
            ``'max'``.
        """
        assert direction in ['max', 'min'], ('The direction parameter should be'
                                             + ' set to either "max" or "min"')

        if self.verbose > 0:
            print("Setting objective")
        if direction == 'max':
            sign = 1
            self.maximize = True
        else:
            sign = -1
            self.maximize = False

        self.reset_objective()
        if objective is None:
            return

        if hasattr(self, 'use_lpi_constraints'):
            if self.use_lpi_constraints and self.verbose > 0:
                warnings.warn("You have the flag `use_lpi_constraints` set to True. Be " +
                              "aware that imposing linearized polynomial constraints will " +
                              "constrain the optimization to distributions with fixed " +
                              "marginals.")

        if (sp.S.One * objective).free_symbols:
            objective = sp.expand(objective)
            symmetrized_objective = {self.One: 0}
            for mon, coeff in objective.as_coefficients_dict().items():
                mon = self._sanitise_monomial(mon)
                symmetrized_objective[mon] = symmetrized_objective.get(mon, 0) + (sign * coeff)
        else:
            symmetrized_objective = {self.One: sign * float(objective)}

        self.objective = symmetrized_objective

        self._update_objective()

    def _update_objective(self):
        """Process the objective with the information from known_moments
        and semiknown_moments.
        """
        self._processed_objective = self.objective.copy()
        known_keys_to_process = set(self.known_moments.keys()).intersection(self._processed_objective.keys())
        known_keys_to_process.discard(self.One)
        for m in known_keys_to_process:
            value = self.known_moments[m]
            self._processed_objective[self.One] += self._processed_objective[m] * value
            del self._processed_objective[m]
        semiknown_keys_to_process = set(self.semiknown_moments.keys()).intersection(self._processed_objective.keys())
        for v1 in semiknown_keys_to_process:
            c1 = self._processed_objective[v1]
            for (k, v2) in self.semiknown_moments[v1]:
                # obj = ... + c1*v1 + c2*v2,
                # v1=k*v2 implies obj = ... + v2*(c2 + c1*k)
                # therefore we add to the coefficient of v2 the term c1*k
                self._processed_objective[v2] = self._processed_objective.get(v2, 0) + c1 * k
                del self._processed_objective[v1]
        # For compatibility purposes
        self._objective_as_name_dict = {k.name: v for (k, v) in self.objective.items()}
        gc.collect(generation=2)  # To reduce memory leaks. Runs after set_values or set_objective.

    def _sanitise_monomial(self, mon: Any, ) -> Union[CompoundMonomial, int]:
        """Bring a monomial into the form used internally.
            NEW: InternalCompoundMonomial are only constructed if in representative form.
            Therefore, if we encounter one, we are good!
        """
        if isinstance(mon, CompoundMonomial):
            return mon
        elif isinstance(mon, (sp.core.symbol.Symbol, sp.core.power.Pow, sp.core.mul.Mul)):
            array = np.concatenate([to_numbers(op, self.names)
                                    for op in flatten_symbolic_powers(mon)])
            return self._sanitise_monomial(array)
        elif isinstance(mon, (tuple, list, np.ndarray)):
            array = np.asarray(mon, dtype=self.np_dtype)
            assert array.ndim == 2, "Cannot allow 1d or 3d arrays as monomial representations."
            assert array.shape[-1] == self._nr_properties, "The input does not conform to the operator specification."
            canon = to_canonical(array, self._notcomm, self._lexorder)
            if mon_is_zero(canon):
                return self.Zero
            else:
                return self.Monomial(canon)
        elif isinstance(mon, str):
            # If it is a string, I assume it is the name of one of the
            # monomials in self.list_of_monomials
            try:
                return self.compound_monomial_from_name_dict[mon]
            except KeyError:
                return self._sanitise_monomial(to_numbers(monomial=mon, parties_names=self.names))
        elif isinstance(mon, numbers.Real):  # If they are number type
            if np.isclose(float(mon), 1):
                return self.One
            elif np.isclose(float(mon), 0):
                return self.Zero
            else:
                raise Exception(f"Constant monomial {mon} can only be 0 or 1.")
        else:
            raise Exception(f"sanitise_monomial: {mon} is of type {type(mon)} and is not supported.")

    def solve(self, interpreter: str = 'MOSEKFusion',
              feas_as_optim: bool = False,
              dualise: bool = True,
              solverparameters=None):
        """Call a solver on the SDP relaxation. Upon successful solution, it
        returns the primal and dual objective values along with the solution
        matrices.

        Parameters
        ----------
        interpreter : str, optional
            The solver to be called. By default ``'MOSEKFusion'``.
        feas_as_optim : bool, optional
            Instead of solving the feasibility problem

                :math:`(1) \text{ find vars such that } \Gamma \succeq 0`

            setting this label to ``True`` solves instead the problem

                :math:`(2) \text{ max }\lambda\text{ such that }
                \Gamma - \lambda\cdot 1 \succeq 0.`

            The correspondence is that the result of (2) is positive if (1) is
            feasible, and negative otherwise. By default ``False``.
        dualise : bool, optional
            Optimize the dual problem (recommended). By default ``True``.
        solverparameters : dict, optional
            Extra parameters to be sent to the solver. By default ``None``.
        """
        if self.momentmatrix is None:
            raise Exception("Relaxation is not generated yet. " +
                            "Call 'InflationSDP.get_relaxation()' first")
        if feas_as_optim and len(self._objective_as_name_dict) > 1:
            warnings.warn("You have a non-trivial objective, but set to solve a " +
                          "feasibility problem as optimization. Setting "
                          + "feas_as_optim=False and optimizing the objective...")
            feas_as_optim = False

        moment_upperbounds = {m.name: val
                              for m, val in self.moment_upperbounds.items()}
        moment_lowerbounds = {**{m.name: val for m, val
                                 in self._processed_moment_lowerbounds.items()},
                              **{m.name: val
                                 for m, val in self.moment_lowerbounds.items()}}
        solveSDP_arguments = {"maskmatrices_name_dict": self.maskmatrices_name_dict,
                              "objective": self._objective_as_name_dict,
                              "known_vars": self.known_moments_name_dict,
                              "semiknown_vars": self.semiknown_moments_name_dict,
                              "feas_as_optim": feas_as_optim,
                              "verbose": self.verbose,
                              "solverparameters": solverparameters,
                              "var_lowerbounds": moment_lowerbounds,
                              "var_upperbounds": moment_upperbounds,
                              "var_equalities": self.moment_linear_equalities,
                              "var_inequalities": self.moment_linear_inequalities,
                              "solve_dual": dualise}

        assert set(self.maskmatrices_name_dict).issuperset(
            set(self.known_moments_name_dict)), 'Error: Assigning known values outside of moment matrix.'

        self.solution_object, lambdaval, self.status = \
            solveSDP_MosekFUSION(**solveSDP_arguments)

        # Process the solution
        if self.status == 'feasible':
            self.primal_objective = lambdaval
            self.objective_value = lambdaval * (1 if self.maximize else -1)

        gc.collect(generation=2)

    ########################################################################
    # PUBLIC ROUTINES RELATED TO THE PROCESSING OF CERTIFICATES            #
    ########################################################################

    def certificate_as_probs(self,
                             clean: bool = False,
                             chop_tol: float = 1e-10,
                             round_decimals: int = 3) -> sp.core.add.Add:
        """Give certificate as symbolic sum of probabilities. The certificate
        of incompatibility is ``cert >= 0``.

        Parameters
        ----------
        clean : bool, optional
            If ``True``, eliminate all coefficients that are smaller than
            ``chop_tol``, normalise and round to the number of decimals
            specified by ``round_decimals``. By default ``False``.
        chop_tol : float, optional
            Coefficients in the dual certificate smaller in absolute value are
            set to zero. By default ``1e-8``.
        round_decimals : int, optional
            Coefficients that are not set to zero are rounded to the number of
            decimals specified. By default ``3``.

        Returns
        -------
        sympy.core.add.Add
            The expression of the certificate in terms or probabilities and
            marginals. The certificate of incompatibility is ``cert >= 0``.
        """
        try:
            dual = self.solution_object['dual_certificate']
        except AttributeError:
            raise Exception("For extracting a certificate you need to solve " +
                            "a problem. Call 'InflationSDP.solve()' first")
        if len(self.semiknown_moments) > 0:
            Warning("Beware that, because the problem contains linearized " +
                    "polynomial constraints, the certificate is not guaranteed " +
                    "to apply to other distributions")
        if clean and not np.allclose(list(dual.values()), 0.):
            dual = clean_coefficients(dual, chop_tol, round_decimals)

        polynomial = sp.S.Zero
        for mon, coeff in dual.items():
            polynomial += coeff * self.name_dict_of_monomials[mon].symbol
        return polynomial


    def certificate_as_string(self,
                              clean: bool = False,
                              chop_tol: float = 1e-10,
                              round_decimals: int = 3) -> sp.core.add.Add:
        """Give the certificate as a string with the notation of the operators
        in the moment matrix.

        Parameters
        ----------
        clean : bool, optional
            If ``True``, eliminate all coefficients that are smaller than
            ``chop_tol``, normalise and round to the number of decimals
            specified by ``round_decimals``. By default ``False``.
        chop_tol : float, optional
            Coefficients in the dual certificate smaller in absolute value are
            set to zero. By default ``1e-8``.
        round_decimals : int, optional
            Coefficients that are not set to zero are rounded to the number of
            decimals specified. By default ``3``.

        Returns
        -------
        str
            The certificate in terms of symbols representing the monomials in
            the moment matrix. The certificate of infeasibility is ``cert > 0``.
        """
        try:
            dual = self.solution_object['dual_certificate']
        except AttributeError:
            raise Exception("For extracting a certificate you need to solve " +
                            "a problem. Call 'InflationSDP.solve()' first")
        if len(self.semiknown_moments) > 0:
            if self.verbose > 0:
                warnings.warn("Beware that, because the problem contains linearized " +
                              "polynomial constraints, the certificate is not guaranteed " +
                              "to apply to other distributions")

        if clean and not np.allclose(list(dual.values()), 0.):
            dual = clean_coefficients(dual, chop_tol, round_decimals)

        rest_of_dual = dual.copy()
        cert_as_string = rest_of_dual.pop('1')
        for mon_name, coeff in rest_of_dual.items():
            cert_as_string += "+" if coeff > 0 else "-"
            cert_as_string += f"{abs(coeff)}*{mon_name}"
        cert_as_string += " >= 0"
        return cert_as_string

    ########################################################################
    # OTHER ROUTINES EXPOSED TO THE USER                                   #
    ########################################################################
    def build_columns(self,
                      column_specification: Union[str, List[List[int]],
                                                  List[sp.core.symbol.Symbol]],
                      max_monomial_length: int = 0,
                      return_columns_numerical: bool = False) -> None:
        """Creates the objects indexing the columns of the moment matrix from
        a specification.

        Parameters
        ----------
        column_specification : Union[str, List[List[int]], List[sympy.core.symbol.Symbol]]
            See description in the ``self.generate_relaxation()`` method.
        max_monomial_length : int, optional
            Maximum number of letters in a monomial in the generating set,
            By default ``0``. Example: if we choose ``'local1'`` for
            three parties, it gives the set :math:`\{1, A, B, C, AB, AC, BC,
            ABC\}`. If we set ``max_monomial_length=2``, the generating set is
            instead :math:`\{1, A, B, C, AB, AC, BC\}`. By default ``0`` (no
            limit).
        return_columns_numerical : bool, optional
            Whether to return the columns also in integer array form (like the
            output of ``to_numbers``). By default ``False``.
        """
        columns = None
        if type(column_specification) == list:
            # There are two possibilities: list of lists, or list of symbols
            if type(column_specification[0]) in {list, np.ndarray}:
                if len(np.array(column_specification[1]).shape) == 2:
                    # This is the format that is later parsed by the program
                    columns = [np.array(mon, dtype=self.np_dtype)
                               for mon in column_specification]
                elif len(np.array(column_specification[1]).shape) == 1:
                    # This is the standard specification for the helper
                    columns = self._build_cols_from_specs(column_specification)
                else:
                    raise Exception('The columns are not specified in a valid format.')
            elif type(column_specification[0]) in [int, sp.Symbol,
                                                   sp.core.power.Pow,
                                                   sp.core.mul.Mul,
                                                   sp.core.numbers.One]:
                columns = []
                for col in column_specification:
                    # We also check the type element by element, and not only the first one
                    if type(col) in [int, sp.core.numbers.One]:
                        if not np.isclose(float(col), 1):
                            raise Exception('The columns are not specified in a valid format.')
                        else:
                            columns += [self.identity_operator]
                    elif type(col) in [sp.Symbol, sp.core.power.Pow, sp.core.mul.Mul]:
                        columns += [to_numbers(str(col), self.names)]
                    else:
                        raise Exception('The columns are not specified in a valid format.')
            else:
                raise Exception('The columns are not specified in a valid format.')
        elif type(column_specification) == str:
            if 'npa' in column_specification.lower():
                npa_level = int(column_specification[3:])
                col_specs = [[]]
                # Determine maximum length
                if (max_monomial_length > 0) and (max_monomial_length < npa_level):
                    max_length = max_monomial_length
                else:
                    max_length = npa_level
                for length in range(1, max_length + 1):
                    for number_tuple in itertools.product(
                            *[range(self.nr_parties)] * length
                    ):
                        a = np.array(number_tuple)
                        # Add only if tuple is in increasing order
                        if np.all(a[:-1] <= a[1:]):
                            col_specs += [a.tolist()]
                columns = self._build_cols_from_specs(col_specs)

            elif 'local' in column_specification.lower():
                local_level = int(column_specification[5:])
                local_length = local_level * self.nr_parties
                # Determine maximum length
                if ((max_monomial_length > 0)
                        and (max_monomial_length < local_length)):
                    max_length = max_monomial_length
                else:
                    max_length = local_length

                party_frequencies = []
                for pfreq in itertools.product(
                        *[range(local_level + 1)] * self.nr_parties
                ):
                    if sum(pfreq) <= max_length:
                        party_frequencies.append(list(reversed(pfreq)))
                party_frequencies = sorted(party_frequencies, key=sum)

                col_specs = []
                for pfreq in party_frequencies:
                    lst = []
                    for party in range(self.nr_parties):
                        lst += [party] * pfreq[party]
                    col_specs += [lst]
                columns = self._build_cols_from_specs(col_specs)

            elif 'physical' in column_specification.lower():
                try:
                    inf_level = int(column_specification[8])
                    length = len(column_specification[8:])
                    message = ("Physical monomial generating set party number" +
                               "specification must have length equal to 1 or " +
                               "number of parties. E.g.: For 3 parties, " +
                               "'physical322'.")
                    assert (length == self.nr_parties) or (length == 1), message
                    if length == 1:
                        physmon_lens = [inf_level] * self.nr_sources
                    else:
                        physmon_lens = [int(inf_level)
                                        for inf_level in column_specification[8:]]
                    max_total_mon_length = sum(physmon_lens)
                except:
                    # If no numbers come after, we use all physical operators
                    physmon_lens = self.inflation_levels
                    max_total_mon_length = sum(physmon_lens)

                if max_monomial_length > 0:
                    max_total_mon_length = max_monomial_length

                party_frequencies = []
                for pfreq in itertools.product(*[range(physmon_lens[party] + 1)
                                                 for party in range(self.nr_parties)]
                                               ):
                    if sum(pfreq) <= max_total_mon_length:
                        party_frequencies.append(list(reversed(pfreq)))
                party_frequencies = sorted(party_frequencies, key=sum)

                physical_monomials = []
                for freqs in party_frequencies:
                    if freqs == [0] * self.nr_parties:
                        physical_monomials.append(self.identity_operator)
                    else:
                        physmons_per_party_per_length = []
                        for party, freq in enumerate(freqs):
                            # E.g., if freq = [1, 2, 0], then
                            # physmons_per_party_per_length will be a list of
                            # lists of physical monomials of length 1, 2 and 0
                            if freq > 0:
                                physmons = phys_mon_1_party_of_given_len(
                                    self.hypergraph,
                                    self.inflation_levels,
                                    party, freq,
                                    self.setting_cardinalities,
                                    self.outcome_cardinalities,
                                    self.names,
                                    self._lexorder)
                                physmons_per_party_per_length.append(physmons)

                        for mon_tuple in itertools.product(
                                *physmons_per_party_per_length):
                            physical_monomials.append(
                                to_canonical(np.concatenate(mon_tuple),
                                             self._notcomm,
                                             self._lexorder))

                columns = physical_monomials
            else:
                raise Exception('I have not understood the format of the '
                                + 'column specification')
        else:
            raise Exception('I have not understood the format of the '
                            + 'column specification')

        if not np.array_equal(self._lexorder, self._default_lexorder):
            res_lexrepr = [nb_mon_to_lexrepr(m, self._lexorder).tolist()
                           if (len(m) or m.shape[-1] == 1) else []
                           for m in columns]
            sortd = sorted(res_lexrepr, key=lambda x: (len(x), x))
            columns = [self._lexorder[lexrepr]
                       if lexrepr != [] else self.identity_operator
                       for lexrepr in sortd]

        columns = [np.array(col, dtype=self.np_dtype).reshape((-1, self._nr_properties)) for col in columns]
        columns_symbolical = [to_symbol(col, self.names) for col in columns]
        if return_columns_numerical:
            return columns_symbolical, columns
        else:
            return columns_symbolical

    def clear_known_values(self) -> None:
        """Clears the information about variables assigned to numerical
        quantities in the problem.
        """
        self.set_values(None)


    def write_to_file(self, filename: str):
        """Exports the problem to a file.

        Parameters
        ----------
        filename : str
            Name of the exported file. If no file format is
            specified, it defaults to sparse SDPA format.
        """
        # Determine file extension
        parts = filename.split('.')
        if len(parts) >= 2:
            extension = parts[-1]
        else:
            extension = 'dat-s'
            filename += '.dat-s'

        # Write file according to the extension
        if self.verbose > 0:
            print('Writing the SDP program to', filename)
        if extension == 'dat-s':
            write_to_sdpa(self, filename)
        elif extension == 'csv':
            write_to_csv(self, filename)
        elif extension == 'mat':
            write_to_mat(self, filename)
        else:
            raise Exception('File format not supported. Please choose between' +
                            ' the extensions .csv, .dat-s and .mat.')

    ########################################################################
    # ROUTINES RELATED TO THE GENERATION OF THE MOMENT MATRIX              #
    ########################################################################
    def _build_cols_from_specs(self, col_specs: List[List[int]]) -> None:
        """Build the generating set for the moment matrix taking as input a
        block specified only the number of parties.

        For example, with col_specs=[[], [0], [2], [0, 2]] as input, we
        generate the generating set S={1, A_{inf}_xa, C_{inf'}_zc,
        A_{inf''}_x'a' * C{inf'''}_{z'c'}} where inf, inf', inf'' and inf'''
        represent all possible inflation copies indices compatible with the
        network structure, and x, a, z, c, x', a', z', c' are all possible input
        and output indices compatible with the cardinalities. As further
        examples, NPA level 2 for three parties is built from
        [[], [0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
        and "local level 1" for three parties is built from
        [[], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]

        Parameters
        ----------
        col_specs : List[List[int]]
            The column specification as specified in the method description.
        """
        if self.verbose > 1:
            # Display col_specs in a more readable way such as 1+A+B+AB etc.
            to_print = []
            for col in col_specs:
                if col == []:
                    to_print.append('1')
                else:
                    to_print.append(''.join([self.names[i] for i in col]))
            print("Column structure:", '+'.join(to_print))

        res = []
        allvars = set()
        for block in col_specs:
            if len(block) == 0:
                res.append(self.identity_operator)
                allvars.add('1')
            else:
                meas_ops = []
                for party in block:
                    meas_ops.append(flatten(self.measurements[party]))
                for monomial_factors in itertools.product(*meas_ops):
                    mon = np.array([to_numbers(op, self.names)[0]
                                    for op in monomial_factors], dtype=self.np_dtype)
                    if self.commuting:
                        canon = remove_projector_squares(mon_lexsorted(mon, self._lexorder))
                        if mon_is_zero(canon):
                            canon = 0
                    else:
                        canon = to_canonical(mon, self._notcomm, self._lexorder)
                    if not np.array_equal(canon, 0):
                        # If the block is [0, 0], and we have the monomial
                        # A**2 which simplifies to A, then A could be included
                        # in the block [0]. We use the convention that [0, 0]
                        # means all monomials of length 2 AFTER simplifications,
                        # so we omit monomials of length 1.
                        if canon.shape[0] == len(monomial_factors):
                            name = to_name(canon, self.names)
                            if name not in allvars:
                                allvars.add(name)
                                if name == '1':
                                    res.append(self.identity_operator)
                                else:
                                    res.append(canon)

        return res

    def _generate_parties(self):
        """Generates all the party operators in the quantum inflation.

        It stores in `self.measurements` a list of lists of measurement
        operators indexed as self.measurements[p][c][i][o] for party p,
        copies c, input i, output o.
        """
        settings = self.setting_cardinalities
        outcomes = self.outcome_cardinalities

        assert len(settings) == len(outcomes), \
            'There\'s a different number of settings and outcomes'
        assert len(settings) == self.hypergraph.shape[1], \
            'The hypergraph does not have as many columns as parties'
        measurements = []
        parties = self.names
        n_states = self.hypergraph.shape[0]
        for pos, [party, ins, outs] in enumerate(zip(parties,
                                                     settings,
                                                     outcomes)):
            party_meas = []
            # Generate all possible copy indices for a party
            all_inflation_indices = itertools.product(
                *[list(range(self.inflation_levels[p_idx]))
                  for p_idx in np.nonzero(self.hypergraph[:, pos])[0]]
            )
            # Include zeros in the positions of states not feeding the party
            all_indices = []
            for inflation_indices in all_inflation_indices:
                indices = []
                i = 0
                for idx in range(n_states):
                    if self.hypergraph[idx, pos] == 0:
                        indices.append('0')
                    elif self.hypergraph[idx, pos] == 1:
                        # The +1 is just to begin at 1
                        indices.append(str(inflation_indices[i] + 1))
                        i += 1
                    else:
                        raise Exception('You don\'t have a proper hypergraph')
                all_indices.append(indices)

            # Generate measurements for every combination of indices.
            # The -1 in outs - 1 is because the use of Collins-Gisin notation
            # (see [arXiv:quant-ph/0306129]), whereby the last operator is
            # understood to be written as the identity minus the rest.
            for indices in all_indices:
                meas = generate_operators(
                    [outs - 1 for _ in range(ins)],
                    party + '_' + '_'.join(indices)
                )
                party_meas.append(meas)
            measurements.append(party_meas)
        self.measurements = measurements

    def _build_momentmatrix(self) -> Tuple[np.ndarray, Dict]:
        """Generate the moment matrix.
        """
        _cols = self.generating_monomials
        problem_arr, canonical_mon_to_idx_dict = calculate_momentmatrix(_cols,
                                                                        self._notcomm,
                                                                        self._lexorder,
                                                                        verbose=self.verbose,
                                                                        commuting=self.commuting,
                                                                        dtype=self.np_dtype)
        idx_to_canonical_mon_dict = {idx: self.to_2dndarray(mon_as_bytes) for (mon_as_bytes, idx) in
                                     canonical_mon_to_idx_dict.items()}

        return problem_arr, idx_to_canonical_mon_dict

    def _calculate_inflation_symmetries(self) -> np.ndarray:
        """Calculates all the symmetries and applies them to the set of
        operators used to define the moment matrix. The new set of operators
        is a permutation of the old. The function outputs a list of all
        permutations.

        Returns
        -------
        List[List[int]]
            The list of all permutations of the generating columns implied by
            the inflation symmetries.
        """

        inflevel = self.inflation_levels
        n_sources = self.nr_sources

        inflation_symmetries = []
        list_original = [self.from_2dndarray(op) for op in self.generating_monomials]
        for source, permutation in tqdm(sorted(
                [(source, permutation) for source in list(range(n_sources))
                 for permutation in itertools.permutations(range(inflevel[source]))]
        ),
                disable=not self.verbose,
                desc="Calculating symmetries       "):
            permutation_plus = np.hstack(([0], np.array(permutation) + 1)).astype(int)
            permuted_cols_ind = \
                apply_source_permutation_coord_input(self.generating_monomials,
                                                     source,
                                                     permutation_plus,
                                                     self.commuting,
                                                     self._notcomm,
                                                     self._lexorder)
            list_permuted = [self.from_2dndarray(op) for op in permuted_cols_ind]
            try:
                total_perm = find_permutation(list_permuted, list_original)
                inflation_symmetries.append(total_perm)
            except:
                if self.verbose > 0:
                    warnings.warn("The generating set is not closed under source swaps." +
                                  "Some symmetries will not be implemented.")

        return np.unique(inflation_symmetries, axis=0)

    def _apply_inflation_symmetries(self,
                                    momentmatrix: np.ndarray,
                                    unsymidx_to_canonical_mon_dict: Dict,
                                    inflation_symmetries: np.ndarray,
                                    conserve_memory=False
                                    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Applies the inflation symmetries to the moment matrix.

        Parameters
        ----------
        momentmatrix : np.ndarray
            The moment matrix.
        unsymidx_to_canonical_mon_dict : Dict
            A dictionary of indices in the moment matrix to their association monomials as 2d numpy arrays.
        inflation_symmetries : List[List[int]]


        It stores in `self.measurements` a list of lists of measurement
        operators indexed as self.measurements[p][c][i][o] for party p,
        copies c, input i, output o.
        """
        unique_values, where_it_matters_flat = np.unique(momentmatrix.flat, return_index=True)
        absent_indices = np.arange(np.min(unique_values))
        symmetric_arr = momentmatrix.copy()

        for permutation in tqdm(inflation_symmetries,
                                disable=not self.verbose,
                                desc="Applying symmetries          "):
            if not np.array_equal(permutation, np.arange(len(momentmatrix))):
                if conserve_memory:
                    for i, ip in enumerate(permutation):
                        for j, jp in enumerate(permutation):
                            new_val = symmetric_arr[i, j]
                            if new_val < symmetric_arr[ip, jp]:
                                symmetric_arr[ip, jp] = new_val
                                symmetric_arr[jp, ip] = new_val
                else:
                    np.minimum(symmetric_arr, symmetric_arr[permutation].T[permutation].T, out=symmetric_arr)
        orbits = np.concatenate((absent_indices, symmetric_arr.flat[where_it_matters_flat].flat))
        # Make the orbits go until the representative
        for key, val in enumerate(orbits):
            previous = 0
            changed = True
            while changed:
                try:
                    val = orbits[val]
                    if val == previous:
                        changed = False
                    else:
                        previous = val
                except KeyError:
                    warnings.warn("Your generating set might not have enough" +
                                  "elements to fully impose inflation symmetries.")
            orbits[key] = val

        old_representative_indices, new_indices, unsym_idx_to_sym_idx = np.unique(orbits,
                                                                                  return_index=True,
                                                                                  return_inverse=True)
        assert np.array_equal(old_representative_indices, new_indices
                              ), 'Something unexpected happened when calculating orbits.'

        symmetric_arr = unsym_idx_to_sym_idx.take(momentmatrix)
        symidx_to_canonical_mon_dict = {new_idx: unsymidx_to_canonical_mon_dict[old_idx] for new_idx, old_idx in
                                        enumerate(
                                            old_representative_indices) if old_idx >= 1}
        return symmetric_arr, unsym_idx_to_sym_idx, symidx_to_canonical_mon_dict

    ########################################################################
    # OTHER ROUTINES                                                       #
    ########################################################################
    def _dump_to_file(self, filename):
        """
        Saves the whole object to a file using `pickle`.

        Parameters
        ----------
        filename : str
            Name of the file.
        """
        import pickle
        with open(filename, 'w') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
