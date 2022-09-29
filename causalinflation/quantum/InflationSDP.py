"""
The module generates the semidefinite program associated to a quantum inflation
instance (see arXiv:1909.10519).

@authors: Alejandro Pozas-Kerstjens, Emanuel-Cristian Boghiu and Elie Wolfe
"""
import gc
import itertools
import warnings
from collections import Counter, deque
from numbers import Real
from typing import List, Dict, Tuple, Union, Any

import numpy as np
import sympy as sp
from scipy.sparse import coo_matrix
from functools import reduce

from causalinflation import InflationProblem
from .fast_npa import (calculate_momentmatrix,
                       to_canonical,
                       to_name,
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
from .writer_utils import write_to_csv, write_to_mat, write_to_sdpa

# Force warnings.warn() to omit the source code line in the message
# https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, line=None:\
    formatwarning_orig(message, category, filename, lineno, line="")
try:
    from tqdm import tqdm
except ImportError:
    from ..utils import blank_tqdm as tqdm


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
        self.hypergraph = np.asarray(self.InflationProblem.hypergraph)
        self.inflation_levels = np.asarray(self.InflationProblem.inflation_level_per_source)
        self._symmetrization_required = np.any(self.inflation_levels - 1)
        if self.supports_problem:
            self.has_children = np.ones(self.nr_parties, dtype=int)
        else:
            self.has_children = self.InflationProblem.has_children
        self.outcome_cardinalities = self.InflationProblem.outcomes_per_party + self.has_children
        self.setting_cardinalities = self.InflationProblem.settings_per_party

        might_have_a_zero = np.any(self.outcome_cardinalities > 1)

        self.measurements = self._generate_parties()
        if self.verbose > 0:
            print("Number of single operator measurements considered per party:", end="")
            prefix = " "
            for i, measures in enumerate(self.measurements):
                counter = itertools.count()
                deque(zip(itertools.chain.from_iterable(itertools.chain.from_iterable(measures)), counter), maxlen=0)
                print(prefix + f"{self.names[i]}={next(counter)}", end="")
                prefix = ", "
            print()
        self.maximize = True  # Direction of the optimization
        self.not_network_model = self.InflationProblem.non_network_scenario
        self._is_knowable_q_non_networks = \
            self.InflationProblem._is_knowable_q_non_networks
        self.rectify_fake_setting = self.InflationProblem.rectify_fake_setting

        self._nr_operators = len(flatten(self.measurements))
        self._nr_properties = 1 + self.nr_sources + 2
        self.np_dtype = np.find_common_type([
            np.min_scalar_type(np.max(self.setting_cardinalities)),
            np.min_scalar_type(np.max(self.outcome_cardinalities)),
            np.min_scalar_type(self.nr_parties + 1),
            np.min_scalar_type(np.max(self.inflation_levels) + 1)], [])
        self.identity_operator = np.empty((0, self._nr_properties), dtype=self.np_dtype)
        self.zero_operator = np.zeros((1, self._nr_properties), dtype=self.np_dtype)

        # Define default lexicographic order through np.lexsort
        # The lexicographic order is encoded as a matrix with rows as
        # operators and the row index gives the order
        arr = np.array([to_numbers(op, self.names)[0]
                        for op in flatten(self.measurements)], dtype=self.np_dtype)
        if might_have_a_zero:
            arr = np.concatenate((self.zero_operator, arr)) # To avoid problems with zero.
        self._default_lexorder = arr[np.lexsort(np.rot90(arr))]
        self._lexorder = self._default_lexorder.copy()

        # Given that most operators commute, we want the matrix encoding the
        # commutations to be sparse, so self._default_commgraph[i, j] = 0
        # implies commutation, and self._default_commgraph[i, j] = 1 is
        # non-commutation.
        self._default_notcomm = notcomm_from_lexorder(self._lexorder, commuting=self.commuting)
        self._notcomm = self._default_notcomm.copy()

        self.canon_ndarray_from_hash_cache = dict()
        self.canonsym_ndarray_from_hash_cache = dict()
        self.atomic_monomial_from_hash_cache = dict()
        self.compound_monomial_from_tuple_of_atoms_cache = dict()
        self.compound_monomial_from_name_dict = dict()
        self.Zero = self.Monomial(self.zero_operator, idx=0)
        self.One = self.Monomial(self.identity_operator, idx=1)

        self._relaxation_has_been_generated = False

    def from_2dndarray(self, array2d: np.ndarray):
        return np.asarray(array2d, dtype=self.np_dtype).tobytes()

    def to_2dndarray(self, bytestream):
        return np.frombuffer(bytestream, dtype=self.np_dtype).reshape((-1, self._nr_properties))

    def to_canonical_memoized(self, array2d: np.ndarray):
        quick_key = self.from_2dndarray(array2d)
        if quick_key in self.canon_ndarray_from_hash_cache:
            return self.canon_ndarray_from_hash_cache[quick_key]
        elif len(array2d) == 0 or np.array_equiv(array2d, 0):
            self.canon_ndarray_from_hash_cache[quick_key] = array2d
            return array2d
        else:
            new_array2d = to_canonical(array2d, self._notcomm, self._lexorder, commuting=self.commuting)
            new_quick_key = self.from_2dndarray(new_array2d)
            self.canon_ndarray_from_hash_cache[quick_key] = new_array2d
            self.canon_ndarray_from_hash_cache[new_quick_key] = new_array2d
            return new_array2d

    def inflation_aware_to_ndarray_representative(self, mon: np.ndarray,
                                                  swaps_plus_commutations=True,
                                                  consider_conjugation_symmetries=False) -> np.ndarray:
        unsym_monarray = self.to_canonical_memoized(mon)
        quick_key = self.from_2dndarray(unsym_monarray)
        if quick_key in self.canonsym_ndarray_from_hash_cache:
            return self.canonsym_ndarray_from_hash_cache[quick_key]
        elif len(unsym_monarray) == 0 or np.array_equiv(unsym_monarray, 0):
            self.canonsym_ndarray_from_hash_cache[quick_key] = unsym_monarray
            return unsym_monarray
        else:
            sym_monarray = to_representative(unsym_monarray,
                                             self.inflation_levels,
                                             self._notcomm,
                                             self._lexorder,
                                             swaps_plus_commutations=swaps_plus_commutations,
                                             consider_conjugation_symmetries=consider_conjugation_symmetries,
                                             commuting=self.commuting)
            self.canonsym_ndarray_from_hash_cache[quick_key] = sym_monarray
            new_quick_key = self.from_2dndarray(sym_monarray)
            if new_quick_key not in self.canonsym_ndarray_from_hash_cache:
                self.canonsym_ndarray_from_hash_cache[new_quick_key] = sym_monarray
            return sym_monarray

    def AtomicMonomial(self, array2d: np.ndarray) -> InternalAtomicMonomial:
        quick_key = self.from_2dndarray(array2d)
        if quick_key in self.atomic_monomial_from_hash_cache:
            return self.atomic_monomial_from_hash_cache[quick_key]
        else:
            #It is important NOT to consider conjugation symmetries for a single factor!
            new_array2d = self.inflation_aware_to_ndarray_representative(array2d,
                                                                         consider_conjugation_symmetries=False)
            new_quick_key = self.from_2dndarray(new_array2d)
            if new_quick_key in self.atomic_monomial_from_hash_cache:
                mon = self.atomic_monomial_from_hash_cache[new_quick_key]
                self.atomic_monomial_from_hash_cache[quick_key] = mon
                return mon
            else:
                mon = InternalAtomicMonomial(inflation_sdp_instance=self, array2d=new_array2d)
                self.atomic_monomial_from_hash_cache[quick_key] = mon
                self.atomic_monomial_from_hash_cache[new_quick_key] = mon
                return mon

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
            return mon
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
        if self.not_network_model:
            minimal_monomial = tuple(tuple(vec) for vec in np.take(atomic_monarray, [0, -2, -1], axis=1))
            return self._is_knowable_q_non_networks(minimal_monomial)
        else:
            return True

    def atomic_knowable_q(self, atomic_monarray: np.ndarray) -> bool:
        first_test = is_knowable(atomic_monarray)
        if not first_test:
            return False
        else:
            return self.inflation_aware_knowable_q(atomic_monarray)


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

    def _operator_max_outcome_q(self, operator: np.ndarray) -> bool:
        """Determines if an operator references the highest possible outcome of a given measurement."""
        party = operator[0] - 1
        return self.has_children[party] and operator[-1] == self.outcome_cardinalities[party] - 2

    def _identify_column_level_equalities(self, generating_monomials):
        """Given the generating monomials, infer implicit equalities between columns of the moment matrix.
        An equality is a dictionary with keys being which column and values being coefficients."""
        column_level_equalities = []
        for i, monomial in enumerate(iter(generating_monomials)):
            for k, operator in enumerate(iter(monomial)):
                if self._operator_max_outcome_q(operator):
                    operator_as_2d = np.expand_dims(operator, axis=0)
                    prefix = monomial[:k]
                    suffix = monomial[(k + 1):]
                    variant_locations = [i]
                    true_cardinality = self.outcome_cardinalities[operator[0] - 1] - 1
                    for outcome in range(true_cardinality - 1):
                        variant_operator = operator_as_2d.copy()
                        variant_operator[0, -1] = outcome
                        variant_monomial = np.vstack((prefix, variant_operator, suffix))
                        for j, monomial in enumerate(iter(generating_monomials)):
                            if np.array_equal(monomial, variant_monomial):
                                variant_locations.append(j)
                                break
                    if len(variant_locations) == true_cardinality:
                        missing_op_location = -1
                        missing_op_monomial = np.vstack((prefix, suffix))
                        for j, monomial in enumerate(self.generating_monomials):
                            if np.array_equal(monomial, missing_op_monomial):
                                missing_op_location = j
                                break
                        if missing_op_location >= 0:
                            column_level_equalities.append((missing_op_location, tuple(variant_locations)))
        num_of_column_level_equalities = len(column_level_equalities)
        if self.verbose > 1 and num_of_column_level_equalities:
            print("Number of column level equalities:", num_of_column_level_equalities)
        return column_level_equalities

    @staticmethod
    def construct_idx_level_equalities_from_column_level_equalities(column_level_equalities,
                                                                    momentmatrix):
        """Given a list of column level equalities (a list of dictionaries with integer keys)
        and the momentmatrix (a ndarray with integer values) we compute the implicit equalities between indices."""
        idx_linear_equalities = []
        seen_already = set()
        for column_level_equality in column_level_equalities:
            for i, row in enumerate(iter(momentmatrix)):
                signature = (i, column_level_equality[-1][-1])
                if signature not in seen_already:
                    seen_already.add(signature)
                    temp_dict = dict()
                    (normalization_col, summation_cols) = column_level_equality
                    norm_idx = row[normalization_col]
                    temp_dict[norm_idx] = 1
                    trivial_count = 0
                    for col in summation_cols:
                        idx = row[col]
                        if idx > 0:  # Nonzero monomial
                            temp_dict[idx] = -1
                        else:
                            trivial_count += 1
                    if trivial_count != 1:
                        idx_linear_equalities.append(temp_dict)
                    del signature, trivial_count, temp_dict
        del seen_already
        return idx_linear_equalities


    ########################################################################
    # MAIN ROUTINES EXPOSED TO THE USER                                    #
    ########################################################################
    def generate_relaxation(self,
                            column_specification:
                            Union[str,
                                  List[List[int]],
                                  List[sp.core.symbol.Symbol]] = 'npa1',
                            suppress_implicit_equalities = False
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
        self.generating_monomials_sym, self.generating_monomials = self.build_columns(column_specification,
                                                                                      return_columns_numerical=True)
        self.nof_columns = len(self.generating_monomials)
        self.column_level_equalities = self._identify_column_level_equalities(self.generating_monomials)

        if self.verbose > 0:
            print("Number of columns:", self.nof_columns)

        # Calculate the moment matrix without the inflation symmetries.
        unsymmetrized_mm_idxs, unsymidx_to_unsym_monarray_dict = self._build_momentmatrix()
        if self.verbose > 0:
            extra_message = (" before symmetrization" if self._symmetrization_required else "")
            print("Number of variables" + extra_message + ":",
                  len(unsymidx_to_unsym_monarray_dict) + (1 if 0 in unsymmetrized_mm_idxs.flat else 0))

        _unsymidx_from_hash_dict = {self.from_2dndarray(v): k for (k, v) in
                                    unsymidx_to_unsym_monarray_dict.items()}

        # Calculate the inflation symmetries.
        self.inflation_symmetries = self._calculate_inflation_symmetries()

        # Apply the inflation symmetries to the moment matrix.
        self.momentmatrix, self.orbits, representative_unsym_idxs = self._apply_inflation_symmetries(unsymmetrized_mm_idxs,
                                                                          self.inflation_symmetries,
                                                                          conserve_memory=False,
                                                                          verbose=self.verbose)
        self.symidx_to_sym_monarray_dict = {self.orbits[unsymidx]: unsymidx_to_unsym_monarray_dict[unsymidx] for unsymidx in representative_unsym_idxs.flat if unsymidx >= 1}
        del unsymmetrized_mm_idxs, unsymidx_to_unsym_monarray_dict
        for (k, v) in _unsymidx_from_hash_dict.items():
            self.canonsym_ndarray_from_hash_cache[k] = self.symidx_to_sym_monarray_dict[self.orbits[v]]
        del _unsymidx_from_hash_dict
        # This is a good time to reclaim memory, as unsymmetrized_mm_idxs can be GBs.
        gc.collect(generation=2)
        (self.momentmatrix_has_a_zero, self.momentmatrix_has_a_one) = np.in1d([0, 1], self.momentmatrix.ravel())
        self.largest_moment_index = max(self.symidx_to_sym_monarray_dict.keys())

        self.compound_monomial_from_idx_dict = dict()
        # The zero monomial is not stored during calculate_momentmatrix, so we manually added it here.
        if self.momentmatrix_has_a_zero:
            self.compound_monomial_from_idx_dict[0] = self.Zero
        for (k, v) in self.symidx_to_sym_monarray_dict.items():
            self.compound_monomial_from_idx_dict[k] = self.Monomial(v, idx=k)

        self.list_of_monomials = list(self.compound_monomial_from_idx_dict.values())
        assert all(v==1 for v in Counter(self.list_of_monomials).values()), "Critical error: multiple indices are being associated to the same monomial."
        knowable_atoms = set()
        for m in self.list_of_monomials:
            knowable_atoms.update(m.knowable_factors)
        self.knowable_atoms = [self.monomial_from_list_of_atomic([atom]) for atom in knowable_atoms]
        del knowable_atoms

        if self.verbose > 0 and self._symmetrization_required:
            print("Number of variables after symmetrization:",
                  len(self.list_of_monomials))

        # Get mask matrices associated with each monomial
        for mon in self.list_of_monomials:
            mon.mask_matrix = coo_matrix(self.momentmatrix == mon.idx).tocsr()
        self.maskmatrices = {mon: mon.mask_matrix for mon in self.list_of_monomials}

        _counter = Counter([mon.knowability_status for mon in self.list_of_monomials])
        self.n_knowable = _counter['Yes']
        self.n_something_knowable = _counter['Semi']
        self.n_unknowable = _counter['No']

        if self.commuting:
            self.possibly_physical_monomials = self.list_of_monomials
        else:
            self.possibly_physical_monomials = [mon for mon in self.list_of_monomials if mon.is_physical]

        # This is useful for the certificates
        self.name_dict_of_monomials = {mon.name: mon for mon in self.list_of_monomials}
        self.monomial_names = list(self.name_dict_of_monomials.keys())

        self.moment_linear_equalities = []
        if suppress_implicit_equalities:
            self.moment_linear_equalities = []
        else:
            self.idx_level_equalities = self.construct_idx_level_equalities_from_column_level_equalities(
                column_level_equalities=self.column_level_equalities,
                momentmatrix=self.momentmatrix)
            self.moment_linear_equalities = [{self.compound_monomial_from_idx_dict[i]: v for i, v in eq.items()}
                                             for eq in self.idx_level_equalities]
            # del idx_level_equalities

        self.moment_linear_inequalities = []
        self.moment_upperbounds = dict()
        self.moment_lowerbounds = {m: 0. for m in self.possibly_physical_monomials}

        self.set_lowerbounds(None)
        self.set_upperbounds(None)
        self.set_objective(None)  # Equivalent to reset_objective
        self.set_values(None)  # Equivalent to reset_values

        self._relaxation_has_been_generated = True

    def reset_objective(self):
        for attribute in {'objective', '_processed_objective',
                          'objective_value', 'primal_objective', 'maximize'}:
            try:
                delattr(self, attribute)
            except AttributeError:
                pass
        self.objective = {self.One: 0.}
        gc.collect(2)

    def reset_values(self):
        self.status = 'not yet solved'
        self.known_moments = dict()
        self.semiknown_moments = dict()
        if self.momentmatrix_has_a_zero:
            self.known_moments[self.Zero] = 0.
        self.known_moments[self.One] = 1.
        gc.collect(2)

    def reset_bounds(self):
        self.reset_lowerbounds()
        self.reset_upperbounds()
        gc.collect(2)

    def reset_lowerbounds(self):
        self.status = 'not yet solved'
        self._processed_moment_lowerbounds = dict()

    def reset_upperbounds(self):
        self.status = 'not yet solved'
        self._processed_moment_upperbounds = dict()

    def update_lowerbounds(self):
        for mon, lb in self.moment_lowerbounds.items():
            self._processed_moment_lowerbounds[mon] = max(self._processed_moment_lowerbounds.get(mon, -np.infty), lb)
        for mon, value in self.known_moments.items():
            try:
                lb = self._processed_moment_lowerbounds[mon]
                assert lb <= value, f"Value {value} assigned for monomial {mon} contradicts the assigned lower bound of {lb}!"
                del self._processed_moment_lowerbounds[mon]
            except KeyError:
                pass

    def update_upperbounds(self):
        for mon, value in self.known_moments.items():
            try:
                ub = self._processed_moment_upperbounds[mon]
                assert ub >= value, f"Value {value} assigned for monomial {mon} contradicts the assigned upper bound of {ub}!"
                del self._processed_moment_upperbounds[mon]
            except KeyError:
                pass

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
        knowable_values = {atom: atom.compute_marginal(prob_array) for atom in self.knowable_atoms} if (prob_array is not None) else dict()
        # Compute self.known_moments and self.semiknown_moments and names their corresponding names dictionaries
        self.set_values(knowable_values, use_lpi_constraints=use_lpi_constraints,
                        only_specified_values=assume_shared_randomness)

    def check_that_known_moments_are_all_knowable(self):
        return all(mon.knowable_q for mon in self.known_moments.keys())

    def set_values(self, values: Union[
        Dict[Union[sp.core.symbol.Symbol, str, CompoundMonomial, InternalAtomicMonomial], float], None],
                   use_lpi_constraints: bool = False,
                   only_specified_values: bool = False) -> None:
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
            Regardless of this flag, unknowable variables can also be fixed
        """

        self.reset_values()

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

        if only_specified_values:
            # If only_specified_values=True, then ONLY the Monomials that
            # are keys in the values dictionary are fixed. Any other monomial
            # that is semi-known relative to the information in the dictionary
            # is left free.
            if self.use_lpi_constraints and self.verbose >= 1:
                warnings.warn(
                    "set_values: Both only_specified_values=True and use_lpi_constraints=True has been detected. " +
                    "With only_specified_values=True, only moments that match exactly " +
                    "those provided in the values dictionary will be set. Values for moments " +
                    "that are products of others moments will not be inferred automatically, " +
                    "and neither will proportionality constraints between moments (LPI constraints). " +
                    "Set only_specified_values=False for these features.")
            self.cleanup_after_set_values()
            return

        # Check for if there are any multi-factors monomials with unspecified-value factors.
        if self.verbose >= 1:
            problematic_multifactor_specified = []
            for mon in self.known_moments.keys():
                if not mon.is_atomic:
                    for atom in mon.factors_as_atomic_monomials:
                        if atom not in self.known_moments:
                            problematic_multifactor_specified.append(mon)
                            break
            if len(problematic_multifactor_specified):
                warnings.warn(
                    "At least one multi-factor monomial has been specified without providing a numerical value for" +
                    "each of its atomic factors:" +
                    f"{problematic_multifactor_specified}"
                )


        atomic_known_moments = {mon.knowable_factors[0]: val for mon, val in self.known_moments.items() if
                                (len(mon) == 1)}
        monomials_not_present_in_moment_matrix = set(self.known_moments.keys()).difference(self.list_of_monomials)
        for mon in monomials_not_present_in_moment_matrix:
            del self.known_moments[mon]

        all_specified_atoms_are_knowable = all(atomic_mon.knowable_q for atomic_mon in atomic_known_moments)
        if all_specified_atoms_are_knowable:
            if not self.use_lpi_constraints:
                remaining_monomials_to_compute = (mon for mon in self.list_of_monomials if
                                                  (not mon.is_atomic) and mon.knowable_q)  # as iterator, saves memory.
            else:
                remaining_monomials_to_compute = (mon for mon in self.list_of_monomials if
                                                  (not mon.is_atomic) and mon.knowability_status in ['Yes',
                                                                                                     'Semi'])  # as iterator, saves memory.
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
                        monomial_corresponding_to_unknown_part = self.monomial_from_list_of_atomic(unknown_atomic_factors)
                        self.semiknown_moments[mon] = (value, monomial_corresponding_to_unknown_part)
                        if self.verbose > 0:
                            if monomial_corresponding_to_unknown_part not in self.list_of_monomials:
                                warnings.warn(
                                    f"Encountered a monomial that does not appear in the original moment matrix:\n {monomial_corresponding_to_unknown_part.name}")
                else:
                    pass
        del atomic_known_moments
        self.cleanup_after_set_values()
        return

    def cleanup_after_set_values(self):
        if self.supports_problem:
            # Convert positive known values into lower bounds.
            nonzero_known_monomials = [mon for mon, value in self.known_moments.items() if not np.isclose(value, 0)]
            for mon in nonzero_known_monomials:
                self._processed_moment_lowerbounds[mon] = 1.
                del self.known_moments[mon]
            self.semiknown_moments = dict()

        # Create lowerbounds list for physical but unknown moments
        self.update_lowerbounds()
        self.update_upperbounds()
        self._update_objective()
        num_nontrivial_known = len(self.known_moments)
        if self.momentmatrix_has_a_zero:
            num_nontrivial_known -= 1
        if self.momentmatrix_has_a_one:
            num_nontrivial_known -= 1
        if self.verbose > 0 and num_nontrivial_known > 0:
            print(f"Number of variables with fixed numeric value: {len(self.known_moments)}")
        num_semiknown = len(self.semiknown_moments)
        if self.verbose > 2 and num_semiknown > 0:
            print(f"Number of semiknown variables: {num_semiknown}")
        return

    def set_objective(self,
                      objective: Union[sp.core.symbol.Symbol, dict, None],
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

        self.reset_objective()
        if direction == 'max':
            self.maximize = True
        else:
            self.maximize = False
        # From a user perspective set_objective(None) should be
        # equivalent to reset_objective()
        if objective is None:
            return
        elif isinstance(objective, sp.core.expr.Expr):
            if objective.free_symbols:
                objective_as_raw_dict = sp.expand(objective).as_coefficients_dict()
                objective_as_raw_dict = {k: float(v) for k, v in objective_as_raw_dict.items()}
            else:
                objective_as_raw_dict = {self.One: float(objective)}
            return self.set_objective(objective_as_raw_dict, direction=direction)
        else:
            if hasattr(self, 'use_lpi_constraints'):
                if self.use_lpi_constraints and self.verbose > 0:
                    warnings.warn("You have the flag `use_lpi_constraints` set to True. Be " +
                                  "aware that imposing linearized polynomial constraints will " +
                                  "constrain the optimization to distributions with fixed " +
                                  "marginals.")
            sign = (1 if self.maximize else -1)
            objective_as_dict = {self.One: 0}
            for mon, coeff in objective.items():
                if not np.isclose(coeff, 0):
                    mon = self._sanitise_monomial(mon)
                    objective_as_dict[mon] = objective_as_dict.get(mon, 0) + (sign * coeff)
            self.objective = objective_as_dict
            if self.supports_problem:
                check_for_uniform_sign = np.array(list(self.objective.values()))
                assert (np.array_equal(check_for_uniform_sign, np.abs(check_for_uniform_sign))
                        or np.array_equal(check_for_uniform_sign, -np.abs(
                            check_for_uniform_sign))), "Cannot evaluate mixed-coefficient objectives for a supports problem."
            self._update_objective()
            return

    def set_upperbounds(self, upperbound_dict: Union[dict, None]) -> None:
        """
        Documentation needed.
        """
        self.reset_upperbounds()
        if upperbound_dict is None:
            return
        sanitized_upperbound_dict = dict()
        for mon, upperbound in upperbound_dict.items():
            mon = self._sanitise_monomial(mon)
            if mon not in sanitized_upperbound_dict.keys():
                sanitized_upperbound_dict[mon] = upperbound
            else:
                old_bound = sanitized_upperbound_dict[mon]
                assert np.isclose(old_bound,
                                  upperbound), f"Contradiction: Cannot set the same monomial {mon} to have different upper bounds."
        self._processed_moment_upperbounds = sanitized_upperbound_dict
        self.update_upperbounds()

    def set_lowerbounds(self, lowerbound_dict: Union[dict, None]) -> None:
        """
        Documentation needed.
        """
        self.reset_lowerbounds()
        if lowerbound_dict is None:
            return
        sanitized_lowerbound_dict = dict()
        for mon, lowerbound in lowerbound_dict.items():
            mon = self._sanitise_monomial(mon)
            if mon not in sanitized_lowerbound_dict.keys():
                sanitized_lowerbound_dict[mon] = lowerbound
            else:
                old_bound = sanitized_lowerbound_dict[mon]
                assert np.isclose(old_bound,
                                  lowerbound), f"Contradiction: Cannot set the same monomial {mon} to have different lower bounds."
        self._processed_moment_lowerbounds = sanitized_lowerbound_dict
        self.update_lowerbounds()

    def set_bounds(self, bounds_dict: Union[dict, None], bound_type: str = 'up') -> None:
        assert bound_type in ['up', 'lo'], ('The bound_type parameter should be'
                                            + ' set to either "up" or "lo"')
        if bound_type == 'up':
            self.set_upperbounds(bounds_dict)
        else:
            self.set_lowerbounds(bounds_dict)

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
        gc.collect(generation=2)  # To reduce memory leaks. Runs after set_values or set_objective.

    def _sanitise_monomial(self, mon: Any, ) -> Union[CompoundMonomial, int]:
        """Bring a monomial into the form used internally.
            NEW: InternalCompoundMonomial are only constructed if in representative form.
            Therefore, if we encounter one, we are good!
        """
        if isinstance(mon, CompoundMonomial):
            return mon
        elif isinstance(mon, (sp.core.symbol.Symbol, sp.core.power.Pow, sp.core.mul.Mul)):
            symbol_to_string_list = flatten_symbolic_powers(mon)
            if len(symbol_to_string_list) == 1:
                try:
                    return self.compound_monomial_from_name_dict[str(symbol_to_string_list[0])]
                except KeyError:
                    pass
            array = np.concatenate([to_numbers(op, self.names)
                                    for op in symbol_to_string_list])
            return self._sanitise_monomial(array)
        elif isinstance(mon, (tuple, list, np.ndarray)):
            array = np.asarray(mon, dtype=self.np_dtype)
            assert array.ndim == 2, "Cannot allow 1d or 3d arrays as monomial representations."
            assert array.shape[-1] == self._nr_properties, "The input does not conform to the operator specification."
            canon = self.to_canonical_memoized(array)
            return self.Monomial(canon) # Automatically adjusts for zero or identity.
        elif isinstance(mon, str):
            # If it is a string, I assume it is the name of one of the
            # monomials in self.list_of_monomials
            try:
                return self.compound_monomial_from_name_dict[mon]
            except KeyError:
                return self._sanitise_monomial(to_numbers(monomial=mon, parties_names=self.names))
        elif isinstance(mon, Real):  # If they are number type
            if np.isclose(float(mon), 1):
                return self.One
            elif np.isclose(float(mon), 0):
                return self.Zero
            else:
                raise Exception(f"Constant monomial {mon} can only be 0 or 1.")
        else:
            raise Exception(f"sanitise_monomial: {mon} is of type {type(mon)} and is not supported.")

    def prepare_solver_arguments(self):
        if self.momentmatrix is None:
            raise Exception("Relaxation is not generated yet. " +
                            "Call 'InflationSDP.get_relaxation()' first")

        assert set(self.known_moments.keys()).issubset(
            self.list_of_monomials), f'Error: Assigning known values outside of moment matrix: {set(self.known_moments.keys()).difference(self.list_of_monomials)}'

        default_return = {"mask_matrices": {mon.name: mon.mask_matrix for mon in self.list_of_monomials},
                    "objective": {m.name: v for m, v in self._processed_objective.items()},
                    "known_vars": {m.name: v for m, v in self.known_moments.items()},
                    "semiknown_vars": {m.name: (v, m2.name) for m, (v, m2) in self.semiknown_moments.items()},
                    "var_equalities": [{m.name: v for m, v in eq.items()} for eq in self.moment_linear_equalities],
                    "var_inequalities": [{m.name: v for m, v in ineq.items()} for ineq in self.moment_linear_inequalities]
                    }
        # Special handling when self.One appears in _processed_moment_lowerbounds or _processed_moment_upperbounds
        if self.One in self.known_moments:
            for m, v in self._processed_moment_lowerbounds.items():
                default_return["var_inequalities"].append({m.name: 1, self.One.name: -v})
            for m, v in self._processed_moment_upperbounds.items():
                default_return["var_inequalities"].append({self.One.name: v, m.name: -1})
        else:
            default_return["known_vars"]['Fake_1'] = 1.
            default_return["mask_matrices"]['Fake_1'] = coo_matrix((self.nof_columns, self.nof_columns)).tocsr()
            for m, v in self._processed_moment_lowerbounds.items():
                default_return["var_inequalities"].append({m.name: 1, 'Fake_1': -v})
            for m, v in self._processed_moment_upperbounds.items():
                default_return["var_inequalities"].append({'Fake_1': v, m.name: -1})
        return default_return


    def solve(self, interpreter: str = 'MOSEKFusion',
              feas_as_optim: bool = False,
              dualise: bool = True,
              solverparameters=None,
              verbose=0,
              core_solver_arguments={}):
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
        verbose : int, optional
            Allows the user to increase the verbosity beyond the InflationSDP.verbose level.
        core_solver_arguments : dict, optional
            By default, solve will use the dictionary of SDP keyword arguments given by prepare_solver_arguments().
            However, a user may manually modify the arguments by passing in their own versions of those keyword arguments here.
        """
        if not self._relaxation_has_been_generated:
            raise Exception("Relaxation is not generated yet. " +
                            "Call 'InflationSDP.get_relaxation()' first")
        if feas_as_optim and len(self._processed_objective) > 1:
            warnings.warn("You have a non-trivial objective, but set to solve a " +
                          "feasibility problem as optimization. Setting "
                          + "feas_as_optim=False and optimizing the objective...")
            feas_as_optim = False

        # When a keyword appears in both prepare_solver_arguments and core_solver_arguments,
        # we use the value manually specified in core_solver_arguments.
        arguments_to_pass_forward = self.prepare_solver_arguments()
        arguments_to_pass_forward.update(core_solver_arguments)
        solveSDP_arguments = {**arguments_to_pass_forward,
                              "feas_as_optim": feas_as_optim,
                              "verbose": max(verbose, self.verbose),
                              "solverparameters": solverparameters,
                              "solve_dual": dualise}

        self.solution_object, lambdaval, self.status = \
            solveSDP_MosekFUSION(**solveSDP_arguments)

        # Process the solution
        if self.status == 'feasible':
            self.primal_objective = lambdaval
            self.objective_value = lambdaval * (1 if self.maximize else -1)
        else:
            self.primal_objective = 'Could not find a value, as the optimization problem was found to be infeasible.'
            self.objective_value = self.primal_objective
        gc.collect(generation=2)

    ########################################################################
    # PUBLIC ROUTINES RELATED TO THE PROCESSING OF CERTIFICATES            #
    ########################################################################

    def certificate_as_probs(self,
                             clean: bool = True,
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
            if clean and np.isclose(int(coeff), round(coeff, round_decimals)):
                coeff = int(coeff)
            polynomial += coeff * self.name_dict_of_monomials[mon].symbol
        return polynomial

    def certificate_as_string(self,
                              clean: bool = True,
                              chop_tol: float = 1e-10,
                              round_decimals: int = 3) -> str:
        """Give the certificate as a string with the notation of the operators
        in the moment matrix.

        Parameters
        ----------
        clean : bool, optional
            If ``True``, eliminate all coefficients that are smaller than
            ``chop_tol``, normalise and round to the number of decimals
            specified by ``round_decimals``. By default ``True``.
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
        if '1' in rest_of_dual:
            if clean:
                cert_as_string = '{0:.{prec}f}'.format(rest_of_dual.pop('1'), prec=round_decimals)
            else:
                cert_as_string = str(rest_of_dual.pop('1'))
        else:
            cert_as_string = ''
        for mon_name, coeff in rest_of_dual.items():
            if mon_name != '0':
                cert_as_string += "+" if coeff >= 0 else ""
                if clean:
                    cert_as_string += '{0:.{prec}f}*{1}'.format(coeff, mon_name, prec=round_decimals)
                else:
                    cert_as_string += f"{abs(coeff)}*{mon_name}"
        cert_as_string += " >= 0"
        if cert_as_string[0] == '+':
            return cert_as_string[1:]
        else:
            return cert_as_string

    ########################################################################
    # OTHER ROUTINES EXPOSED TO THE USER                                   #
    ########################################################################
    def build_columns(self,
                      column_specification: Union[str, List[List[int]],
                                                  List[sp.core.symbol.Symbol]],
                      max_monomial_length: int = 0,
                      return_columns_numerical: bool = False):
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
                                self.to_canonical_memoized(np.concatenate(mon_tuple)))
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
                    canon = self.to_canonical_memoized(mon)
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
        return measurements

    def _build_momentmatrix(self) -> Tuple[np.ndarray, Dict]:
        """Generate the moment matrix.
        """
        problem_arr, canonical_mon_as_bytes_to_idx_dict = calculate_momentmatrix(self.generating_monomials,
                                                                        self._notcomm,
                                                                        self._lexorder,
                                                                        verbose=self.verbose,
                                                                        commuting=self.commuting,
                                                                        dtype=self.np_dtype)
        idx_to_canonical_mon_dict = {idx: self.to_2dndarray(mon_as_bytes) for (mon_as_bytes, idx) in
                                     canonical_mon_as_bytes_to_idx_dict.items()}
        del canonical_mon_as_bytes_to_idx_dict
        return problem_arr, idx_to_canonical_mon_dict

    def _calculate_inflation_symmetries(self, generators_only=True) -> np.ndarray:
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

        symmetry_inducing_sources = [source for source, inf_level in enumerate(self.inflation_levels) if inf_level > 1]
        # inflevel = self.inflation_levels
        # n_sources = self.nr_sources
        if len(symmetry_inducing_sources):
            inflation_symmetries = []
            identity_permutation_of_columns = np.arange(self.nof_columns, dtype=int)
            list_original = [self.from_2dndarray(op) for op in self.generating_monomials]
            for source in tqdm(symmetry_inducing_sources,
                               disable=not self.verbose,
                               desc="Calculating symmetries...    "):
                inflation_symmetries_from_this_source = [identity_permutation_of_columns]
                for permutation in itertools.permutations(range(self.inflation_levels[source])):
                    # We do NOT need to calculate the "symmetry" induced by the identity permutation.
                    if not permutation == tuple(range(self.inflation_levels[source])):
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
                            inflation_symmetries_from_this_source.append(np.asarray(total_perm, dtype=int))
                        except:
                            if self.verbose > 0:
                                warnings.warn("The generating set is not closed under source swaps." +
                                              "Some symmetries will not be implemented.")
                inflation_symmetries.append(inflation_symmetries_from_this_source)
            if generators_only:
                inflation_symmetries_flat = list(itertools.chain.from_iterable((perms[1:] for perms in inflation_symmetries)))
                return np.unique(inflation_symmetries_flat, axis=0)
            else:
                inflation_symmetries_flat = [reduce(np.take, perms) for perms in itertools.product(*inflation_symmetries)]
                return np.unique(inflation_symmetries_flat[1:], axis=0)
        else:
            return np.empty((0, len(self.generating_monomials)), dtype=int)

    @staticmethod
    def _apply_inflation_symmetries(momentmatrix: np.ndarray,
                                    inflation_symmetries: np.ndarray,
                                    conserve_memory=False,
                                    verbose=0
                                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        if not len(inflation_symmetries):
            default_array = np.arange(momentmatrix.max() + 1)
            return momentmatrix, default_array, default_array
        else:
            unique_values, where_it_matters_flat = np.unique(momentmatrix.flat, return_index=True)
            absent_indices = np.arange(np.min(unique_values))
            symmetric_arr = momentmatrix.copy()

            for permutation in tqdm(inflation_symmetries,
                                    disable=not verbose,
                                    desc="Applying symmetries...       "):
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
            return symmetric_arr, unsym_idx_to_sym_idx, old_representative_indices

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
