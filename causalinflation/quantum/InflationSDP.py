# import copy
# for i in set(globals().keys()):
#     if not i.startswith('_'):
#         exec('del ' + i)

import gc
import itertools
import operator

import numpy as np
import pickle
import sympy as sp
from collections import Counter  # , defaultdict, namedtuple

# from numpy import ndarray

from causalinflation import InflationProblem
from causalinflation.quantum.general_tools import (to_representative,
                                                   to_numbers,
                                                   to_symbol,
                                                   flatten,
                                                   flatten_symbolic_powers,
                                                   phys_mon_1_party_of_given_len,
                                                   is_knowable,
                                                   find_permutation,
                                                   apply_source_permutation_coord_input,
                                                   from_numbers_to_flat_tuples,
                                                   generate_noncommuting_measurements,
                                                   clean_coefficients
                                                   )
from causalinflation.quantum.fast_npa import (calculate_momentmatrix,
                                              to_canonical,
                                              to_name,
                                              remove_projector_squares,
                                              mon_lexsorted,
                                              mon_is_zero,
                                              nb_mon_to_lexrepr,
                                              nb_commuting)

import sys
# def try_del(attribute: str, context=sys.modules[__name__]):
#     try:
#         delattr(context, attribute)
#     except AttributeError:
#         pass
# for attr in {'AtomicMonomial', 'CompoundMonomial', 'AtomicMonomialMeta', 'CompoundMonomialMeta'}:
#     try_del(attr)

from causalinflation.quantum.monomial_class import (AtomicMonomial, CompoundMonomial, to_tuple_of_tuples)
from causalinflation.quantum.monomial_class import Monomial as preMonomial
from causalinflation.quantum.sdp_utils import solveSDP_MosekFUSION
from causalinflation.quantum.writer_utils import (write_to_csv, write_to_mat,
                                                  write_to_sdpa)

from causalinflation.quantum.types import List, Dict, Tuple, Union, Any
# from typing import List, Dict, Union, Tuple, Any
import warnings

# Force warnings.warn() to omit the source code line in the message
# Source: https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
    formatwarning_orig(message, category, filename, lineno, line='')
# from warnings import warn

from scipy.sparse import coo_matrix  # , dok_matrix

try:
    from numba import types
    from numba.typed import Dict as nb_Dict
    from tqdm import tqdm
except ImportError:
    def tqdm(*args, **kwargs):
        return args[0]


class InflationSDP(object):
    """Class for generating and solving an SDP relaxation for quantum inflation.

    Parameters
    ----------
    inflationproblem : InflationProblem
        Details of the scenario.
    commuting : bool, optional
        Whether variables in the problem are going to be commuting
        (classical problem) or non-commuting (quantum problem),
        by default False.
    verbose : int, optional
        Optional parameter for level of verbose:
            * 0: quiet (default),
            * 1: verbose,
            * 2: debug level,
        by default 0.
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
        self.InflationProblem = inflationproblem  # Worth storing?
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

        (self.measurements, self.substitutions, self.names) = self._generate_parties()

        self.maximize = True  # Direction of the optimization
        self.split_node_model = self.InflationProblem.split_node_model
        self.is_knowable_q_split_node_check = self.InflationProblem.is_knowable_q_split_node_check
        self.rectify_fake_setting_atomic_factor= self.InflationProblem.rectify_fake_setting_atomic_factor

        self._nr_operators = len(flatten(self.measurements))
        self._nr_properties = 1 + self.nr_sources + 2

        # Define default lexicographic order through np.lexsort
        # The lexicographic order is encoded as a matrix with rows as 
        # operators and the row index gives the order
        arr = np.array([to_numbers(op, self.names)[0]
                        for op in flatten(self.measurements)], dtype=np.uint16)
        self._default_lexorder = arr[np.lexsort(np.rot90(arr))]
        # self._PARTY_ORDER = 1 + np.array(list(range(self.nr_parties)))
        self._lexorder = self._default_lexorder.copy()

        # Given that most operators commute, we want the matrix encoding the
        # commutations to be sparse, so self._default_commgraph[i, j] = 0
        # implies commutation, and self._default_commgraph[i, j] = 1 is
        # non-commutation.
        self._default_notcomm = np.zeros((self._lexorder.shape[0],
                                          self._lexorder.shape[0]), dtype=int)
        for i in range(self._lexorder.shape[0]):
            for j in range(i, self._lexorder.shape[0]):
                if i == j:
                    self._default_notcomm[i, j] = 0
                else:
                    self._default_notcomm[i, j] = int(not nb_commuting(self._lexorder[i],
                                                                       self._lexorder[j]))
                    self._default_notcomm[j, i] = self._default_notcomm[i, j]
        self._notcomm = self._default_notcomm.copy()  # ? Ideas for a better name?

        # TODO: This seems invariant? How we do alter the lexorder?
        self.just_inflation_indices = np.array_equal(self._lexorder,
                                                     self._default_lexorder)  # Use lighter version of to_rep
        AtomicMonomial.reset_all()
        CompoundMonomial.reset_all()

    def commutation_relationships(self):
        """This returns a user-friendly representation of the commutation relationships."""
        from collections import namedtuple
        nonzero = namedtuple('NonZeroExpressions', 'exprs')
        data = []
        for i in range(self._lexorder.shape[0]):
            for j in range(i, self._lexorder.shape[0]):
                # Most operators commute as they belong to different parties,
                # so it is more interested to list those that DONT'T commute.
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

    # TODO Low priority future feature, currently not important, but don't delete
    # # def set_custom_lexicographic_order(self, 
    # #                                    custom_lexorder: Dict[sp.Symbol, int],
    # #                                    ) -> None:
    # #     if custom_lexorder is not None:
    # #         assert isinstance(custom_lexorder, dict), \
    # #                 "custom_lexicographic_order must be a dictionary"

    # #         ### First process the values
    # #         # If the user gives lex ranks such as 1, 5, 3, 7, sort them, and
    # #         # reindex them from 0 to the number of them: 1, 5, 3, 7 -> 0, 2, 1, 3.
    # #         # This way, the lex rank also is useful for indexing a matrix.
    # #         v = sorted(list(custom_lexorder.values()))
    # #         assert len(np.unique(np.array(v))) == len(v), "Lex ranks must be unique"
    # #         v_old_to_new = {v: i for i, v in enumerate(v)}
    # #         custom_lexorder = dict(zip(custom_lexorder.keys(),
    # #                                 [v_old_to_new[v]
    # #                                 for v in custom_lexorder.values()]))
    # #         ### Now process the keys
    # #         lexorder = np.zeros((self._nr_operators, self._nr_properties),
    # #                             dtype=np.uint16)
    # #         for key, value in custom_lexorder.items():
    # #             if type(key) in [sp.Symbol, sp.core.power.Pow, sp.core.mul.Mul]:
    # #                 array = to_numbers(str(key), self.names)
    # #             elif type(key) == Monomial:
    # #                 array = Monomial.as_ndarray.astype(np.uint16)
    # #             else:
    # #                 raise Exception(f"_nb_process_lexorder: Key type {type(key)} not allowed.")
    # #             assert len(array) == 1, "Cannot assign lex rank to a product of operators."
    # #             lexorder[value, :] = array[0]
    # #         self._lexorder = lexorder

    # #         ### Now some consistency checks
    # #         # 1. Check that the lex order is a permutation of the default lex order
    # #         assert set(to_tuple_of_tuples(self._lexorder)) \
    # #                 == set(to_tuple_of_tuples(self._default_lexorder)), \
    # #                 "Custom lexicographic order does not contain the correct operators."

    # #         # 2. Check if ops for the same party are together forming a 
    # #         # continuous block
    # #         custom_sorted_parties = self._lexorder[:, 0]
    # #         past_i = set([custom_sorted_parties[0]])
    # #         i_old = custom_sorted_parties[0]
    # #         for i in custom_sorted_parties[1:]:
    # #             if i != i_old:
    # #                 if i not in past_i:
    # #                     past_i.add(i)
    # #                 else:
    # #                     warnings.warn("WARNING: Custom lexicographic order is does not " + 
    # #                                   "order parties in continuous blocks. " +
    # #                                   "This affects functionality such as identifying zero monomials " +
    # #                                   "due to products of orthogonal operators corresponding to " + 
    # #                                   "different outputs. It is strongly recommended to " +
    # #                                   "order parties in continuous blocks where operators with all " +
    # #                                   "else equal except the outputs are grouped together.")
    # #                     break
    # #             i_old = i

    # #         # 3. Check if the order of the contiguous blocks of parties is
    # #         # consistent with the names argument
    # #         custom_parties, _ = nb_unique(custom_sorted_parties)
    # #         custom_party_names = [self.names[i - 1] for i in custom_parties]
    # #         if custom_party_names != self.names:
    # #             if self.verbose > 0:
    # #                 warnings.warn("Custom lexicographic order orders 'names' " + 
    # #                                f"as {custom_party_names} whereas the previous value " +
    # #                                f"was {self.names}. This affects functionality such as " +
    # #                                "setting values and distributions.")
    # #             # self.names = custom_party_names # TODO what happens with names??
    # #             # self._PARTY_ORDER = custom_parties 

    # #     else:
    # #         self._lexorder =  self._default_lexorder
    # #         # self._PARTY_ORDER = 1 + np.array(list(range(self.nr_parties)))

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

    # TODO: Write an assertion that all (naturally) atomic monomials are now mapped to themselves under to_representative.
    # TODO: Emi will write a test?
    def inflation_aware_to_ndarray_representative(self, mon: np.ndarray,
                                                  swaps_plus_commutations=True,
                                                  consider_conjugation_symmetries=True) -> np.ndarray:
        unsym_monarray = to_canonical(mon, self._notcomm, self._lexorder)

        try:
            sym_monarray = self.unsym_monarray_to_sym_monarray[to_tuple_of_tuples(unsym_monarray)]
        except KeyError:
            # warnings.warn(
            #     f"Encountered a monomial that does not appear in the original moment matrix: {unsym_monarray}")
            sym_monarray = to_representative(unsym_monarray,
                                              self.inflation_levels,
                                              self._notcomm,
                                              self._lexorder,
                                              swaps_plus_commutations=swaps_plus_commutations,
                                              consider_conjugation_symmetries=consider_conjugation_symmetries,
                                              commuting=self.commuting)
            hashable_sym_monarray = to_tuple_of_tuples(sym_monarray)
            if hashable_sym_monarray not in self.unsym_monarray_to_sym_monarray:
                warnings.warn(
                    f"Encountered a monomial that does not appear in the original moment matrix:\n {unsym_monarray}")
            self.unsym_monarray_to_sym_monarray[hashable_sym_monarray] = sym_monarray
        self.unsym_monarray_to_sym_monarray[to_tuple_of_tuples(sym_monarray)] = sym_monarray
        return sym_monarray

    def inflation_aware_to_representative(self, *args, **kwargs) -> Tuple[Tuple]:
        return to_tuple_of_tuples(self.inflation_aware_to_ndarray_representative(*args, **kwargs))

    def sanitise_compoundmonomial(self, mon: CompoundMonomial) -> CompoundMonomial:
        # Sanity check if need be.
        # for atom in mon.factors_as_atomic_monomials:
        #     assert atom.as_ndarray.shape[-1] == self._nr_properties, f"Somehow we have screwed up the monomial storage! {atom.as_ndarray} from {mon.as_ndarray}"
        mon.update_atomic_constituents(self.inflation_aware_to_ndarray_representative,
                                       just_inflation_indices=True)  # MOST IMPORTANT
        # More sanity checking, if needed.
        # for atom in mon.factors_as_atomic_monomials:
        #     assert atom.as_ndarray.shape[-1] == self._nr_properties, f"Somehow we have screwed up the monomial storage! {atom.as_ndarray} from {mon.as_ndarray}"
        #     assert (atom.inflation_indices_are_irrelevant or not atom.not_yet_updated_by_to_representative), f"Hang on, all monomials should have been set to representative by construction! {atom.as_ndarray} from {mon.as_ndarray}"

        mon.update_rectified_arrays_based_on_fake_setting_correction(
            self.rectify_fake_setting_atomic_factor)
        mon.update_name_and_symbol_given_observed_names(self.names)
        return mon

    def Monomial(self,
                 array2d: np.ndarray,
                 sandwich_positivity=True,
                 idx=-1) -> CompoundMonomial:
        obj = preMonomial(array2d,
                                                          atomic_is_knowable=self.atomic_knowable_q,
                                                          sandwich_positivity=sandwich_positivity,
                                                          idx=idx)
        assert isinstance(obj, CompoundMonomial), 'CompoundMonomial failed to be generated!'
        return self.sanitise_compoundmonomial(obj)

    #

    def generate_relaxation(self,
                            column_specification:
                            Union[str,
                                  List[List[int]],
                                  List[sp.core.symbol.Symbol]] = 'npa1'
                            ) -> None:
        """Creates the SDP relaxation of the quantum inflation problem
        using the NPA hierarchy and applies the symmetries inferred
        from inflation.

        It takes as input the generating set of monomials {M_i}_i. The moment
        matrix Gamma is defined by all the possible inner products between these
        monomials:

        .. math::
            \Gamma[i, j] := \operatorname{tr} (\rho * (M_i)^\dagger M_j).

        The set {M_i} is specified by the parameter `column_specification`.

        In the inflated graph there are many symmetries coming from invariance
        under swaps of the copied sources, which are used to remove variables
        in the moment matrix.

        Parameters

        column_specification : Union[str, List[List[int]], List[sympy.core.symbol.Symbol]]
            Describes the generating set of monomials {M_i}_i.

            (NOTATION) If we have 2 parties, we denote by {A, B} the set
            of all measurement operators of these two parties. That is, {A, B}
            represents {A_{InflIndices1}_x_a, B_{InflIndices2}_y_b} for all
            possible indices {InflIndices1, InflIndices2, x, a, y, b}.
            Similarly, the product {A*B} represents the product of the
            operators of A and B for all possible indices. Note that with this
            notation, A*A and A**2 represent different sets.

            * `(str) 'npaN'`: where N is an integer. This represents NPA level N.
            This is built by taking the cartesian product of the flattened
            set of measurement operators N times and removing duplicated
            elements. For example, level 3 with measurements {A, B} will give
            the set {1, A, B, A*A, A*B, B*B, A*A*A, A*A*B, A*B*C, A*B*B,
            B*B*B}. This is known to converge to the quantum set Q for
            N->\infty.

            * `(str) 'localN'`: where N is an integer. This gives a subset of NPA level N+1.
            Local level N considers monomials that have at most
            N measurement operators per party. For example, `local1` is a
            subset of `npa2`; for 2 parties, `npa2` is {1, A, B, A*A, A*B, B*B}
            while `local1` is {1, A, B, A*B}. Note that terms such as
            A*A are missing as that is more than N=1 measurements per party.

            * `(str) 'physicalN'`: The subset of local level N with only all commuting operators.
            We only consider commutation coming from having different supports.
            `N` cannot be greater than the smallest number of copies of a source
            in the inflated graph. For example, in the bilocal scenario
            A-source-B-source-C with 2 outputs and no inputs, `physical2` only
            gives 5 possibilities for Bob: {1, B_1_1_0_0, B_2_2_0_0,
            B_1_1_0_0*B_2_2_0_0,  B_1_2_0_0*B_2_1_0_0}. There are no other
            products where all operators commute. The full set of physical
            generating monomials is built by taking the cartesian product
            between all possible physical monomials of each party.

            * `List[List[int]]`: This encodes a party block structure.
            Each integer encodes a party. Within a party block, all missing
            input, output and inflation  indices are taken into account.
            For example, [[], [0], [1], [0, 1]] gives the set {1, A, B, A*B},
            which is the same as 'local1'. The set [[], [0], [1], [2], [0, 0],
            [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]] is the same as {1, A, B,
            C, A*A, A*B, A*C, B*B, B*C, C*C} which is the same as 'npa2' for
            3 parties. [[]] encodes the identity element.

            * `List[sympy.core.symbol.Symbol]`: we can fully specify the generating set by
            giving a list of symbolic operators built from the measurement
            operators in `self.measurements`. This list needs to have the
            identity `sympy.S.One` as the first element.
        """
        # self.use_lpi_constraints = False

        # Process the column_specification input and store the result
        # in self.generating_monomials.
        self.generating_monomials_sym, self.generating_monomials = \
            self.build_columns(column_specification,
                               return_columns_numerical=True)

        if self.verbose > 0:
            print("Number of columns:", len(self.generating_monomials))

        # Calculate the moment matrix without the inflation symmetries.
        self.unsymmetrized_mm_idxs, self.unsymidx_to_unsym_monarray_dict = self._build_momentmatrix()
        if self.verbose > 1:
            print("Number of variables before symmetrization:",
                  len(self.unsymidx_to_unsym_monarray_dict))

        # for monarray in self.unsymidx_to_unsym_monarray_dict.values():
        #     assert np.asarray(monarray).shape[-1] == self._nr_properties, f"Somehow we have screwed up the monomial storage! {mon.as_ndarray}"

        self.unsym_monarray_to_unsymidx_dict = {to_tuple_of_tuples(v): k for (k, v) in
                                                self.unsymidx_to_unsym_monarray_dict.items()}

        # Calculate the inflation symmetries.
        self.inflation_symmetries = self._calculate_inflation_symmetries()

        # Apply the inflation symmetries to the moment matrix.
        self.momentmatrix, self.orbits, self.symidx_to_sym_monarray_dict \
            = self._apply_inflation_symmetries(self.unsymmetrized_mm_idxs,
                                               self.unsymidx_to_unsym_monarray_dict,
                                               self.inflation_symmetries)
        self.unsym_monarray_to_sym_monarray = {k: self.symidx_to_sym_monarray_dict[self.orbits[v]] for (k, v) in
                                               self.unsym_monarray_to_unsymidx_dict.items()}

        self.largest_moment_index = max(self.symidx_to_sym_monarray_dict.keys())

        # ZeroMon = Monomial([[]], idx=0)
        # ZeroMon.name = '0'
        # ZeroMon.mask_matrix =
        # sample_monomial = np.asarray(self.symidx_to_canonical_mon_dict[2], dtype=np.uint16)
        #
        self.One = self.Monomial(np.empty((0, self._nr_properties), dtype=np.uint16), idx=1)
        # Emi: I think it makes sense to have this,
        # as we reference this in other places and its a pain
        # to redefine it
        self.just_inflation_indices = np.array_equal(self._lexorder,
                                                     self._default_lexorder)  # Use lighter version of to_rep
        self.list_of_monomials = [self.One] + [self.Monomial(v, idx=k)
                                               for (k, v) in self.symidx_to_sym_monarray_dict.items()]
        for mon in self.list_of_monomials:
            mon.mask_matrix = coo_matrix(self.momentmatrix == mon.idx).tocsr()

        # #COMMENT ELIE TO EMI: Since we are using an inflation-aware Monomial constructor, all this processing is taken care of!
        # self.set_of_atomic_monomials = set()
        # for mon in self.list_of_monomials:
        #     self.set_of_atomic_monomials.update(mon.factors_as_atomic_monomials)


        # # set(itertools.chain.from_iterable((mon.factors_as_atomic_monomials for mon in self.list_of_monomials)))
        # if not self.just_inflation_indices:
        #     # If we have a custom lexorder, then it might be that the user
        #     # chooses A_2_0_2 to be lower than A_1_0_1. There are compelling reasons
        #     # to allow this for a general NPO program (e.g., if we want to
        #     # implement that the product of two operators is zero, and they
        #     # don't appear together in the canonical order, we could change
        #     # the order to make them be one next to the other, then pass it
        #     # through to_canonical and see if they end up together, and this
        #     # would mean the monomial is zero). However, for inflation, there
        #     # is no benefit, so instead of changing the function to_representative
        #     # to adapt to arbitrary lex order, I will just pass all monomials
        #     # through the standard to_representative.
        #     for atomic_mon in self.set_of_atomic_monomials:
        #         atomic_mon.update_hash_via_to_representative_function(self.inflation_aware_to_ndarray_representative)
        #     self.__factor_reps_computed__ = True  # So they are not computed again
        # else:
        #     self.__factor_reps_computed__ = False
        #
        # self.idx_dict_of_monomials = {mon.idx: mon for mon in
        #                               sorted(self.list_of_monomials, key=operator.attrgetter('idx'))}
        # # self._all_atomic_knowable = set()
        # for atomic_mon in self.set_of_atomic_monomials:
        #     atomic_mon.update_rectified_array_based_on_fake_setting_correction(
        #         self.rectify_fake_setting_atomic_factor)
        #
        # for mon in self.list_of_monomials:
        #     """
        #     Assigning each monomial a meaningful name. ONLY CALL AFTER RECTIFY SETTINGS!
        #     """
        #     mon.update_name_and_symbol_given_observed_names(observable_names=self.names)

        """
        Used only for internal diagnostics.
        """
        _counter = Counter([mon.knowability_status for mon in self.list_of_monomials])
        self._n_knowable = _counter['Yes']
        self._n_something_knowable = _counter['Semi']
        self._n_unknowable = _counter['No']

        if self.commuting:
            self.possibly_physical_monomials = self.list_of_monomials
        else:
            self.possibly_physical_monomials = [mon for mon in self.list_of_monomials if mon.physical_q]

        # This is useful for the certificates
        self.name_dict_of_monomials = {mon.name: mon for mon in self.list_of_monomials}
        # Note indexing starts from zero, for certificate compatibility. #TODO: Does it, though?
        self.monomial_names = list(self.name_dict_of_monomials.keys())

        self.maskmatrices_name_dict = {mon.name: mon.mask_matrix for mon in self.list_of_monomials}
        # self.maskmatrices_idx_dict = {mon.idx: mon.mask_matrix for mon in self.list_of_monomials}
        self.maskmatrices = {mon: mon.mask_matrix for mon in self.list_of_monomials}

        self.moment_linear_equalities = []
        self.moment_linear_inequalities = []

        self.set_objective(None)  # Equivalent to reset_objective
        self.set_values(None)  # Equivalent to reset_values

        # Elie comment: these are not used anywhere.
        # _counter = Counter([mon.known_status for mon in self.list_of_monomials if mon.idx > 0])
        # self._n_known = _counter['Yes']
        # self._n_something_known = _counter['Semi']
        # self._n_unknown = _counter['No']

        # Hack to avoid calculating the representative factors unless needed
        # They are needed if we want to set values of unknowable moments

    def reset_objective(self):
        self.objective = {self.One: 0}
        self._objective_as_name_dict = {'1': 0.}
        # self._objective_as_idx_dict = {1: 0.}

    def reset_values(self):
        # for mon in self.list_of_monomials:
        #     for attribute in {'unknown_part', 'known_status', 'known_value', 'unknown_signature'}:
        #         try:
        #             delattr(mon, attribute)
        #         except AttributeError:
        #             pass
        # if mon.idx > 1:
        #     mon.known_status = 'No'
        # elif mon.idx == 1:
        #     mon.known_status = 'Yes'
        # mon.known_value = 1.
        # mon.unknown_part = mon.as_ndarray
        for attribute in {'known_moments', 'nof_known_moments', 'semiknown_moments', 'moment_upperbounds',
                          'known_moments_name_dict', 'semiknown_moments_name_dict', 'moment_upperbounds_name_dict'}:
            try:
                delattr(self, attribute)
            except AttributeError:
                pass
        gc.collect(2)
        self.known_moments = {self.One: 1.}
        self.nof_known_moments = len(self.known_moments)
        self.semiknown_moments = dict()
        self.moment_upperbounds = dict()
        # TODO: REMOVE ALL REFERENCES TO NAME DICTS
        self.known_moments_name_dict = {'1': 1.}
        self.semiknown_moments_name_dict = dict()
        self.moment_upperbounds_name_dict = dict()
        # self.known_moments_idx_dict = {1: 1.}
        # self.semiknown_moments_idx_dict  = dict()
        # self.moment_upperbounds_idx_dict = dict()
        self.reset_physical_lowerbounds()

    def reset_physical_lowerbounds(self):
        self.physical_monomials = set(self.possibly_physical_monomials).difference(self.known_moments.keys())
        self.moment_lowerbounds = {mon: 0. for mon in self.physical_monomials}
        # BELOW TO BE DEPRECATED
        self.physical_monomial_names = set(mon.name for mon in self.physical_monomials)
        self.moment_lowerbounds_name_dict = {name: 0 for name in self.physical_monomial_names}
        # self.physical_monomial_idxs = set(mon.idx for mon in self.physical_monomials)
        # self.moment_lowerbounds_idx_dict = {idx: 0. for idx in self.physical_monomial_idxs}

    def update_physical_lowerbounds(self):
        self.physical_monomials = set(self.possibly_physical_monomials).difference(self.known_moments.keys())
        for mon in self.physical_monomials.difference(self.moment_lowerbounds.keys()):
            self.moment_lowerbounds[mon] = 0.
        for mon in self.physical_monomials.intersection(self.moment_lowerbounds.keys()):
            self.moment_lowerbounds[mon] = max(0., self.moment_lowerbounds[mon])
        # BELOW TO BE DEPRECATED
        self.physical_monomial_names = set(mon.name for mon in self.physical_monomials)
        for name in self.physical_monomial_names.difference(self.moment_lowerbounds_name_dict.keys()):
            self.moment_lowerbounds_name_dict[name] = 0.
        for name in self.physical_monomial_names.intersection(self.moment_lowerbounds_name_dict.keys()):
            self.moment_lowerbounds_name_dict[name] = max(0., self.moment_lowerbounds_name_dict[name])
        # self.physical_monomial_idxs = set(mon.idx for mon in self.physical_monomials)
        # for idx in self.physical_monomial_idxs.difference(self.moment_lowerbounds_idx_dict.keys()):
        #     self.moment_lowerbounds_idx_dict[idx] = 0.
        # for idx in self.physical_monomial_idxs.intersection(self.moment_lowerbounds_idx_dict.keys()):
        #     self.moment_lowerbounds_idx_dict[idx] = max(0., self.moment_lowerbounds_idx_dict[idx])

    # def set_distribution(self,
    #                      prob_array: Union[np.ndarray, None],
    #                      use_lpi_constraints: bool = False,
    #                      treat_as_support = False) -> None:
    #     """Set numerically the knowable moments and semiknowable moments according
    #     to the probability distribution specified. If p is None, or the user
    #     doesn't pass any argument to set_distribution, then this is understood
    #     as a request to delete information about past distributions.
    #     Args:
    #         prob_array (np.ndarray): Multidimensional array encoding the
    #         distribution, which is called as prob_array[a,b,c,...,x,y,z,...]
    #         where a,b,c,... are outputs and x,y,z,... are inputs.
    #     Note: even if the inputs have cardinality 1, they must be specified,
    #         and the corresponding axis dimensions are 1.

    #         use_lpi_constraints (bool): Specification whether linearized
    #         polynomial constraints (see, e.g., Eq. (D6) in arXiv:2203.16543)
    #         will be imposed or not.
    #     """

    #     self.reset_distribution()

    #     if prob_array is None:
    #         # From a user perspective set_distribution(None) should be
    #         # equivalent to reset_distribution()
    #         return

    #     self.use_lpi_constraints = use_lpi_constraints

    #     if (len(self._objective_as_idx_dict) > 1) and self.use_lpi_constraints:
    #         warnings.warn("You have an objective function set. Be aware that imposing " +
    #              "linearized polynomial constraints will constrain the " +
    #              "optimization to distributions with fixed marginals.")

    #     hashable_prob_array = to_tuple_of_tuples(prob_array)
    #     dict_which_groups_monomials_by_representative = defaultdict(list)
    #     for mon in self.list_of_monomials:
    #         mon.update_given_prob_dist(hashable_prob_array)
    #         if mon.known_status == 'Semi':
    #             if not self.use_lpi_constraints:
    #                 mon.known_status = 'No'
    #                 mon.unknown_part = mon.as_ndarray
    #             elif mon.known_value == 0:
    #                 mon.known_value = 0
    #                 mon.known_status = 'Yes'
    #                 mon.unknown_part = tuple()
    #         mon.representative = self.inflation_aware_to_representative(mon.unknown_part)
    #         dict_which_groups_monomials_by_representative[mon.representative].append(mon)

    #     """
    #     This next block of code re-indexes the monomials (and the momentmatrix) 
    #     to put the known variables first, then the unknown, then the semiknown.
    #     """
    #     # known_statusus = np.empty((self.largest_moment_index + 1,), dtype='<U4')
    #     # known_statusus[[0, 1]] = 'Yes'
    #     # for monomial in self.list_of_monomials:
    #     #     known_statusus[monomial.idx] = monomial.known_status

    #     _counter = Counter([mon.known_status for mon in self.list_of_monomials if mon.idx > 0])
    #     self._n_known = _counter['Yes']
    #     self._n_something_known = _counter['Semi']
    #     self._n_unknown = _counter['No']

    #     # #REORDERING TO KEEP KNOWN UP FRONT
    #     # _reordering_of_monomials = np.argsort(np.concatenate((
    #     #     np.flatnonzero(known_statusus == 'Yes'),
    #     #     np.flatnonzero(known_statusus == 'No'),
    #     #     np.flatnonzero(known_statusus == 'Semi'))))
    #     # self.momentmatrix = _reordering_of_monomials.take(self.momentmatrix)
    #     # self.orbits = _reordering_of_monomials.take(self.orbits)
    #     # for mon in self.list_of_monomials:
    #     #     mon.idx = _reordering_of_monomials[mon.idx]

    #     self._semiknowns_without_counterparts = set()
    #     if self.use_lpi_constraints:
    #         for representative, list_of_mon in dict_which_groups_monomials_by_representative.items():
    #             if any(mon.known_status == 'Semi' for mon in list_of_mon):
    #                 which_is_wholly_unknown = [mon.known_status == 'No' for mon in list_of_mon]
    #                 if not np.count_nonzero(which_is_wholly_unknown) >= 1:
    #                     self._semiknowns_without_counterparts.add(representative)
    #                 # NEXT SIX LINES ARE FOR LEGACY COMPATABILITY
    #                 list_of_mon_copy = list_of_mon.copy()
    #                 list_of_mon_copy = sorted(list_of_mon_copy, key=operator.attrgetter('known_value'))
    #                 big_val_mon = list_of_mon_copy.pop(-1)
    #                 # wholly_unknown_mon = list_of_mon_copy.pop(np.flatnonzero(which_is_wholly_unknown)[0])
    #                 for semiknown_mon in list_of_mon_copy:
    #                     coeff = np.true_divide(semiknown_mon.known_value,big_val_mon.known_value)
    #                     self.semiknown_moments_idx_dict[semiknown_mon.idx] = (coeff, big_val_mon.idx)
    #                     self.semiknown_moments_name_dict[semiknown_mon.name] = (coeff, big_val_mon.name)
    #         if self.verbose and len(self._semiknowns_without_counterparts):
    #             warning_string = f'We found {len(self._semiknowns_without_counterparts)} semiknowns with no counterparts:'
    #             for representative in self._semiknowns_without_counterparts:
    #                 warning_string += ('\n' + np.array_str(np.asarray(representative)))
    #             formatwarning_orig = warnings.formatwarning
    #             warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
    #                 formatwarning_orig(message, category, filename, lineno, line='')
    #             warnings.warn(warning_string)

    #         max_semiknown_coefficient = max(coeiff for (coeiff, idx) in self.semiknown_moments_idx_dict.values())
    #         # max(max(mon.known_value for mon in list_of_mon)
    #         #                                 for v in dict_which_groups_monomials_by_representative.values())
    #         assert max_semiknown_coefficient <= 1, f'Some semi-expressible-coefficient exceeds one: {max_semiknown_coefficient}'

    #     #RESET PROPERTIES

    #     for mon in self.list_of_monomials:
    #         if mon.known_status == 'Yes':
    #             if treat_as_support and mon.known_value > 0:
    #                 self.moment_lowerbounds_idx_dict[mon.idx] = 1
    #                 self.moment_lowerbounds_name_dict[mon.name] = 1
    #             else:
    #                 self.known_moments_idx_dict[mon.idx] = mon.known_value
    #                 self.known_moments_name_dict[mon.name] = mon.known_value
    #     self.nof_known_moments = len(self.known_moments_idx_dict)
    #     #Such as resetting physical_monomials using new indices.
    #     if not self.commuting:
    #         self.physical_monomial_idxs = set([mon.idx for mon in self.list_of_monomials if mon.physical_q]).difference(self.known_moments_idx_dict.keys())
    #         self.physical_monomial_names = set([mon.name for mon in self.list_of_monomials if mon.physical_q]).difference(
    #             self.known_moments_name_dict.keys())
    #         self.moment_lowerbounds_idx_dict = {physical_idx: 0 for physical_idx in self.physical_monomial_idxs}
    #         self.moment_lowerbounds_name_dict = {physical_name: 0 for physical_name in self.physical_monomial_names}

    #     if self.objective and not (prob_array is None):
    #         warnings.warn('Danger! User apparently set the objective before the distribution.')
    #     self.distribution_has_been_set = True

    # TODO: Shouldn't we set use_lpi_constraints=True by default?

    def set_distribution(self,
                         prob_array: Union[np.ndarray, None],
                         use_lpi_constraints: bool = False) -> None:
        """Set numerically the knowable moments and semiknowable moments according
        to the probability distribution specified. If p is None, or the user
        doesn't pass any argument to set_distribution, then this is understood
        as a request to delete information about past distributions.
        Args:
            prob_array (np.ndarray): Multidimensional array encoding the
            distribution, which is called as prob_array[a,b,c,...,x,y,z,...]
            where a,b,c,... are outputs and x,y,z,... are inputs.
        Note: even if the inputs have cardinality 1, they must be specified,
            and the corresponding axis dimensions are 1.

            use_lpi_constraints (bool): Specification whether linearized
            polynomial constraints (see, e.g., Eq. (D6) in arXiv:2203.16543)
            will be imposed or not.
        """
        # Reset is performed by set_values
        # self.reset_distribution()

        if prob_array is None:
            # From a user perspective set_distribution(None) should be
            # equivalent to reset_distribution(), i.e., set_values({})
            values = dict()
        else:
            # The monomial class can compute its own known and unknown parts given atomic valuations.

            # atomic_knowable_CompoundMonomials = [m for m in self.list_of_monomials
            #                    if m.nof_factors == 1 and m.knowability_status == 'Yes']

            knowable_values = {m: m.compute_marginal(prob_array) for m in self.list_of_monomials
                               if m.nof_factors == 1 and m.knowability_status == 'Yes'}

        # Compute self.known_moments and self.semiknown_moments and names their corresponding names dictionaries
        self.set_values(knowable_values, use_lpi_constraints=use_lpi_constraints,
                        only_knowable_moments=True)

        # if self.objective and not (prob_array is None):
        #     warnings.warn('Danger! User apparently set the objective before the distribution.')
        # self.distribution_has_been_set = True

    def set_values(self, values: Union[
        Dict[Union[sp.core.symbol.Symbol, str, CompoundMonomial, AtomicMonomial], float], None],
                   use_lpi_constraints: bool = False,
                   normalised: bool = True,
                   only_knowable_moments: bool = True,
                   only_specified_values: bool = False,
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
            
        set_unknowable_moments : bool
            Specifies whether the user wishes to also set the values of 
            monomials that are not a priori knowable.
            
        normalised: bool
            Specifies whether the unit monomial '1' is given value 1.0 even if
            '1' is not included in the values dictionary (default, True), or if 
            is left as a free variable (False).
        """

        # TODO: Note to Elie from Emi. I have given some thought about this and  
        # I wrote this in two ways: one where the user can specify unknowable
        # moments and one where they cannot. In the first I need to find
        # the representatives of ALL factors, and in the other I can use
        # the functions you wrote and find the representative only of the
        # semiknown unknowable factors.
        # An example of fixing unknown monomials could be (me) trying to study
        # the impact of nonconvex constraints by fixing one of the variables. 
        # E.g., relax v=v1*v2 by fixing v1 and letting v2 be 
        # free and then doing a sweep in the values of v1.

        self.reset_values()
        if (values is None) or (len(values) == 0):
            # From a user perspective set_values(None) should be
            # equivalent to reset_distribution().
            return

        self.use_lpi_constraints = use_lpi_constraints

        if (len(self.objective) > 1) and self.use_lpi_constraints:
            warnings.warn("You have an objective function set. Be aware that imposing " +
                          "linearized polynomial constraints will constrain the " +
                          "optimization to distributions with fixed marginals.")

        # Sanitise the values dictionary
        self.known_moments = {self._sanitise_monomial(k): v for (k, v) in values.items() if not np.isnan(v)}
        # # values_clean = dict()
        # for k in list(values.keys()):
        #     k_sanitised = self._sanitise_monomial(k)
        #     #NOTE Elie to Emi: We need to KEEP the fact that these are equal to zero until after we work out the semiknowns.
        #     if not np.array_equal(k_sanitised, 0):
        #         values_clean[k_sanitised] = values[k]
        #     else:
        #         warnings.warn("The variable " + k + " in the values provided " +
        #                     "simplifies to 0 with the current substitution rules.")

        # Check that the keys are consistent with the flags set
        for k in self.known_moments:
            if only_specified_values is False:
                if k.nof_factors > 1:
                    raise Exception("set_values: The monomial " + str(k) + " is not an " +
                                    "atomic monomial, but composed of several factors. " +
                                    "Please provide values only for atomic monomials. " +
                                    "If you want to manually be able to set values for " +
                                    "non-atomic monomials, set only_specified_values to True.")
            if k.knowability_status != 'Yes' and only_knowable_moments is True:
                raise Exception("set_values: You are trying to set the value of " + str(k) +
                                "which is not fully knowable in standard scenarios. " +
                                "Set set_only_knowable_moments to False for this feature.")

        if normalised:
            self.known_moments[self.One] = 1  # Automatic in reset_values. #TODO: Should this not be always?

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

        # Some more pre-processing!
        # Elie to Emi: We need to call to representative on everything ONLY in order to handle semiknowns.
        # if (self.use_lpi_constraints or not only_knowable_moments) and not self.__factor_reps_computed__:
        #     for atomic_mon in self.set_of_atomic_monomials:
        #         atomic_mon.update_hash_via_to_representative_function(self.inflation_aware_to_ndarray_representative)
        #     # for mon in self.list_of_monomials:
        #     #     # For the representantive we use the monomial self-hashing feature.
        #     #     mon.update_atomic_constituents(self.inflation_aware_to_representative)
        #     #     mon.update_hash_via_to_representative_function(self.inflation_aware_to_representative)
        #     #     # mon.as_ndarray = self.inflation_aware_to_representative(mon.as_ndarray)
        #     #     # mon._factors_repr = [to_tuple_of_tuples(  # So that it is hashable
        #     #     #                         self.inflation_aware_to_representative(factor)
        #     #     #                         )
        #     #     #                     for factor in mon.factors]
        #     self.__factor_reps_computed__ = True  # So they are not computed again

        # atomic_known_moments = {mon.knowable_factors[0]: val for mon, val in self.known_moments.items() if (len(mon) == 1)}
        if only_knowable_moments:
            remaining_monomials_to_compute = (mon for mon in self.list_of_monomials if
                                              len(mon) > 1 and mon.knowable_q)  # as iterator, saves memory.
        else:
            remaining_monomials_to_compute = (mon for mon in self.list_of_monomials if len(mon) > 1)

        for mon in remaining_monomials_to_compute:
            value, unknown_CompoundMonomial, known_status = mon.evaluate_given_atomic_monomials_dict(self.known_moments,
                                                                                                     use_lpi_constraints=self.use_lpi_constraints)
            # assert isinstance(value, float), f'expected numeric value! {value}'
            if known_status == 'Yes':
                self.known_moments[mon] = value
            elif known_status == 'Semi':
                # if self.use_lpi_constraints:
                #     if np.isclose(value, 0):
                #         self.known_moments[mon] = 0
                #     else:
                # unknown_CompoundMonomial.update_atomic_constituents(
                #     to_representative_function=self.inflation_aware_to_ndarray_representative,
                #     just_inflation_indices=self.just_inflation_indices
                # )
                self.semiknown_moments[mon] = (value, self.sanitise_compoundmonomial(unknown_CompoundMonomial))
                # assert isinstance(self.semiknown_moments, dict)

            else:
                pass
        # del atomic_known_moments
        # del remaining_monomials_to_compute
        gc.collect(generation=2)
        #
        #
        # # self.semiknown_moments = dict() # Already reset by reset_distribution
        # if only_knowable_moments:
        #     # Use the Monomial methods developed by Elie that rely on knowability status
        #     for mon in filter(lambda x: x.nof_factors > 1, self.list_of_monomials):
        #         valuation = [self.atomic_known_moments.get(f, default=np.nan) for f in mon.knowable_factors]
        #         #if mon.knowability_status != 'No':
        #         # valuation = [values_clean_as_knowabletuple[f] if f in values_clean_as_knowabletuple else np.nan
        #         #                 for f in mon.knowable_factors]
        #         value, unknown_CompoundMonomial = mon.evaluate_given_valuation_of_knowable_part(valuation)
        #         if mon.known_status == 'Yes':
        #             self.known_moments[mon] = value
        #         elif mon.known_status == 'Semi':
        #             self.semiknown_moments[mon] = (value, unknown_CompoundMonomial)
        #             # reprr = self.inflation_aware_to_representative(
        #             #                 to_canonical(mon.unknown_part.astype(np.uint16), self._notcomm, self._lexorder))
        #             # unknown = Monomial(reprr, atomic_is_knowable=self.atomic_knowable_q,
        #             #                         sandwich_positivity=True)
        #             # unknown.update_name_and_symbol_given_observed_names(self.names)
        #             # self.semiknown_moments[mon] = (mon.known_value, unknown)
        #         else:
        #             pass
        # else:
        #     # We don't use Elie's Monomial methods, as those rely on knowability status.
        #     # Here, the knowability status of an atomic monomial is simply
        #     # whether it is in the values dictionary or not. This allows the user
        #     # to also set unknowable monomials.
        #     for mon in filter(lambda x: x.nof_factors > 1, self.list_of_monomials):
        #         unknowns_to_join = []
        #         known = 1
        #         for i, atomic_mon in enumerate(mon.factors_as_atomic_monomials):
        #             if atomic_mon in values_clean:
        #                 known *= values_clean[atomic_mon]
        #             else:
        #                 unknowns_to_join.append(mon.factors[i])
        #         if len(unknowns_to_join) == 0:
        #             self.known_moments[mon] = known
        #         else:
        #             if self.use_lpi_constraints:
        #                 if len(unknowns_to_join) < mon.nof_factors:
        #                     if np.isclose(known, 0):
        #                         self.known_moments[mon] = 0
        #                     else:
        #                         unknown = Monomial(self.inflation_aware_to_ndarray_representative(
        #                                                     to_canonical(np.concatenate(unknowns_to_join),
        #                                                                  self._notcomm, self._lexorder)),
        #                                                     atomic_is_knowable=self.atomic_knowable_q,
        #                                                     sandwich_positivity=True)
        #                         unknown.update_name_and_symbol_given_observed_names(self.names)
        #                         self.semiknown_moments[mon] = (known, unknown)

        self.cleanup_after_set_values(use_lpi_constraints=self.use_lpi_constraints)
        return

    def cleanup_after_set_values(self, use_lpi_constraints=False):
        # if use_lpi_constraints:
        #     for mon, (value, unknown) in self.semiknown_moments.items():
        #         # assert isinstance(value, float), f'expected numeric value! {value}'
        #         if np.isclose(value, 0):
        #             del self.semiknown_moments[mon]
        #             self.known_moments[mon] = 0

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

            # TODO: ADD EQUALITY CONSTRAINTS FOR SUPPORTS PROBLEM!

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
            Direction of the optimization (max/min), by default 'max'
        """

        assert direction in ['max', 'min'], ('The direction parameter should be'
                                             + ' set to either "max" or "min"')
        if direction == 'max':
            sign = 1
            self.maximize = True
        else:
            sign = -1
            self.maximize = False

        self.reset_objective()
        # From a user perspective set_objective(None) should be
        # equivalent to reset_objective()
        if objective is None:
            return

        if hasattr(self, 'use_lpi_constraints'):
            if self.use_lpi_constraints:
                warnings.warn("You have the flag `use_lpi_constraints` set to True. Be " +
                              "aware that imposing linearized polynomial constraints will " +
                              "constrain the optimization to distributions with fixed " +
                              "marginals.")

        if (sp.S.One * objective).free_symbols:
            objective = sp.expand(objective)
            symmetrized_objective = {self.One: 0}  # Used for updated with known monomials.
            # symmetrized_objective = dict()
            for mon, coeff in objective.as_coefficients_dict().items():
                Mon = self._sanitise_monomial(mon)
                symmetrized_objective[Mon] = symmetrized_objective.get(Mon, 0) + (sign * coeff)
                # if Mon in symmetrized_objective:
                #     symmetrized_objective[Mon] += sign * coeff
                # else:
                #     symmetrized_objective[Mon] = sign * coeff
        else:
            symmetrized_objective = {self.One: sign * float(objective)}

        self.objective = symmetrized_objective

        self._update_objective()

    def _update_objective(self):
        """Process the objective with the information from known_moments
        and semiknown_moments.
        """
        if list(self.objective.keys()) != [self.One]:
            self._processed_objective = self.objective.copy()
            for m, value in self.known_moments.items():
                if m != self.One and m in self._processed_objective:
                    self._processed_objective[self.One] += self._processed_objective[m] * value
                    del self._processed_objective[m]
            if hasattr(self, 'use_lpi_constraints'):
                if self.use_lpi_constraints:
                    for v1, (k, v2) in self.semiknown_moments.items():
                        # obj = ... + c1*v1 + c2*v2,
                        # v1=k*v2 implies obj = ... + v2*(c2 + c1*k)
                        # therefore we need to add to the coefficient of v2 the term c1*k 
                        if v1 in self._processed_objective:
                            c1 = self._processed_objective[v1]
                            if v2 in self._processed_objective:
                                self._processed_objective[v2] += c1 * k
                            else:
                                self._processed_objective[v2] = c1 * k
                            del self._processed_objective[v1]

            # For compatibility purposes
            self._objective_as_name_dict = {k.name: v for (k, v) in self._processed_objective.items()}

    def _sanitise_monomial(self, mon: Any) -> Union[CompoundMonomial, int]:
        """Bring a monomial into the form used internally.
        """
        if type(mon) in [sp.core.symbol.Symbol, sp.core.power.Pow, sp.core.mul.Mul, sp.Symbol]:  # Elie comment: should not be sp.Symbol
            # This assumes the monomial is in "machine readable symbolic" form
            array = np.concatenate([to_numbers(op, self.names)
                                    for op in flatten_symbolic_powers(mon)])
            assert array.ndim == 2, "Cannot allow 3d or 1d arrays as monomial representations."
            assert array.shape[-1] == self._nr_properties, "Something is wrong with the to_numbers usage."
        elif type(mon) in [tuple, list]:
            array = np.array(mon, dtype=np.uint16)
            assert array.ndim == 2, "Cannot allow 1d or 3d arrays as monomial representations."
            assert array.shape[-1] == self._nr_properties, "The input does not conform to the operator specification."
        elif type(mon) == str:
            # If it is a string, I assume it is the name of one of the 
            # monomials in self.list_of_monomials
            if hasattr(self, 'list_of_monomials'):
                for m in self.list_of_monomials:
                    if m.name == mon:
                        return m
                raise Exception(f"sanitise_monomial: {mon} in string format " +
                                "is not found in `list_of_monomials`.")
            else:
                raise Exception(f"sanitise_monomial: string format as input " +
                                "is not supported before the generation of " +
                                "`list_of_monomials`.")
        elif isinstance(mon, CompoundMonomial):
            array = mon.as_ndarray
            if hasattr(self, 'list_of_monomials'):
                # If the user passes a proper Monomial, if they instantiated it
                # themselves, it can be in a form not found in list_of_monomials.
                if mon in self.list_of_monomials:
                    return mon
        else:  # If they are number type
            try:
                if np.isclose(float(mon), 1):
                    return self.One
                else:
                    raise Exception(f"Constant monomial {mon} can only be the identity monomial.")
            except:
                # This can happen if calling float() gives an error
                Exception(f"sanitise_monomial: {mon} is of type {type(mon)} and is not supported.")

        canon = to_canonical(array, self._notcomm, self._lexorder)
        if np.array_equal(canon, 0):
            return 0  # TODO: Or ZeroMonomial once that is implemented?
        else:
            # canon = self.inflation_aware_to_ndarray_representative(canon)  # Not needed, I think.
            reprr = self.Monomial(canon)
            # reprr.update_atomic_constituents(to_representative_function=self.inflation_aware_to_ndarray_representative,
            #                                  just_inflation_indices=self.just_inflation_indices)
            # reprr.update_rectified_arrays_based_on_fake_setting_correction(
            #     self.InflationProblem.rectify_fake_setting_atomic_factor)
            # reprr.update_name_and_symbol_given_observed_names(self.names)
            return reprr

    # TODO: I'd like to add the ability to handle 4 classes of problem: SAT, CERT, OPT, SUPP
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
            The solver to be called, by default 'MOSEKFusion'.
        feas_as_optim : bool, optional
            Instead of solving the feasibility problem
                (1) find vars such that Gamma >= 0
            setting this label to True solves instead the problem
                (2) max lambda such that Gamma - lambda*Id >= 0.
            The correspondence is that the result of (2) is positive if (1) is
            feasible and negative otherwise. By default False.
        solverparameters : dict, optional
            Extra parameters to be sent to the solver, by default None.

        """
        if self.momentmatrix is None:
            raise Exception("Relaxation is not generated yet. " +
                            "Call 'InflationSDP.get_relaxation()' first")
        # if not self.distribution_has_been_set:
        #     self.set_distribution(prob_array=None, use_lpi_constraints=False)
        if feas_as_optim and len(self._objective_as_name_dict) > 1:
            warnings.warn("You have a non-trivial objective, but set to solve a " +
                          "feasibility problem as optimization. Setting "
                          + "feas_as_optim=False and optimizing the objective...")
            feas_as_optim = False

        solveSDP_arguments = {"maskmatrices_name_dict": self.maskmatrices_name_dict,
                              "objective": self._objective_as_name_dict,
                              "known_vars": self.known_moments_name_dict,
                              "semiknown_vars": self.semiknown_moments_name_dict,
                              "positive_vars": self.physical_monomial_names,
                              "feas_as_optim": feas_as_optim,
                              "verbose": self.verbose,
                              "solverparameters": solverparameters,
                              "var_lowerbounds": self.moment_lowerbounds_name_dict,
                              "var_upperbounds": self.moment_upperbounds_name_dict,
                              "var_equalities": self.moment_linear_equalities,
                              "var_inequalities": self.moment_linear_inequalities,
                              "solve_dual": dualise}

        assert set(self.maskmatrices_name_dict).issuperset(set(self.known_moments_name_dict)), 'Error: Assigning known values outside of moment matrix.'

        self.solution_object, lambdaval, self.status = \
            solveSDP_MosekFUSION(**solveSDP_arguments)

        # Process the solution
        if self.status == 'feasible':
            self.primal_objective = lambdaval
            self.objective_value = lambdaval * (1 if self.maximize else -1)

        if self.status in ['feasible', 'infeasible']:
            self.dual_certificate = self.solution_object['dual_certificate']

    def certificate_as_probs(self, clean: bool = False,
                             chop_tol: float = 1e-10,
                             round_decimals: int = 4) -> sp.core.symbol.Symbol:
        """Give certificate as symbolic sum of probabilities that is greater
        than or equal to 0 for feasible distributions.

        Parameters
        ----------
        clean : bool, optional
            If true, eliminate all coefficients that are smaller
            than 'chop_tol' and round to the number of decimals specified
            `round_decimals`. Defaults to True.
        chop_tol : float, optional
            Coefficients in the dual certificate smaller in absolute value are
            set to zero. Defaults to 1e-8.
        round_decimals : int, optional
            Coefficients that are not set to zero are rounded to the number
            of decimals specified. Defaults to 3.

        Returns
        -------
        sympy.core.symbol.Symbol
            The certificate in terms or probabilities and marginals.
        """

        try:
            cert = self.dual_certificate
        except AttributeError:
            raise Exception("For extracting a certificate you need to solve " +
                            "a problem. Call 'InflationSDP.solve()' first")
        if len(self.semiknown_moments_name_dict) > 0:  # TODO update this if we discard the names dict
            string = ("Beware that, because the problem contains linearized " +
                      "polynomial constraints, the certificate is not guaranteed " +
                      "to apply to other distributions")
            Warning(string)

        mons = [self.name_dict_of_monomials[name] for name in cert.keys()]
        coeffs = np.array(list(cert.values()))
        if clean and not np.allclose(coeffs, 0):
            coeffs = clean_coefficients(coeffs, chop_tol, round_decimals)
        cert = dict(zip(mons, coeffs))

        polynomial = 0
        for mon, coeff in cert.items():
            polynomial += coeff * mon.to_symbol()

        return sp.expand(polynomial)

    def certificate_as_objective(self, clean: bool = False,
                                 chop_tol: float = 1e-10,
                                 round_decimals: int = 3) -> sp.core.symbol.Symbol:
        """Give certificate as symbolic sum of operators that can be used
        as an objective function to optimse.
    
        Parameters
        ----------
        clean : bool, optional
            If true, eliminate all coefficients that are smaller
            than 'chop_tol', normalise and round to the number of decimals
            specified `round_decimals`. Defaults to True.
        chop_tol : float, optional
            Coefficients in the dual certificate smaller in absolute value are
            set to zero. Defaults to 1e-8.
        round_decimals : int, optional
            Coefficients that are not set to zero are rounded to the number
            of decimals specified. Defaults to 3.
    
        Returns
        -------
        sympy.core.symbol.Symbol
            The certificate as an objective function.
        """
        try:
            cert = self.dual_certificate
        except AttributeError:
            raise Exception("For extracting a certificate you need to solve " +
                            "a problem. Call 'InflationSDP.solve()' first")
        if len(self.semiknown_moments_name_dict) > 0:  # TODO update this if we discard the names dict
            string = ("Beware that, because the problem contains linearized " +
                      "polynomial constraints, the certificate is not guaranteed " +
                      "to apply to other distributions")
            Warning(string)

        mons = [self.name_dict_of_monomials[name] for name in cert.keys()]
        coeffs = np.array(list(cert.values()))
        if clean and not np.allclose(coeffs, 0):
            coeffs = clean_coefficients(coeffs, chop_tol, round_decimals)
        cert = dict(zip(mons, coeffs))

        polynomial = 0
        for mon, coeff in cert.items():
            polynomial += coeff * mon.to_symbol(objective_compatible=True)

        return sp.expand(polynomial)

    def certificate_as_correlators(self,
                                   clean: bool = False,
                                   chop_tol: float = 1e-10,
                                   round_decimals: int = 3,
                                   use_langlerangle=False) -> sp.core.symbol.Symbol:
        """Give certificate as symbolic sum of 2-output correlators that
        is greater than or equal to 0. Only valid for 2-output problems.
    
        Parameters
        ----------
        clean : bool, optional
            If true, eliminate all coefficients that are smaller
            than 'chop_tol', normalise and round to the number of decimals
            specified `round_decimals`. Defaults to True.
        chop_tol : float, optional
            Coefficients in the dual certificate smaller in absolute value are
            set to zero. Defaults to 1e-8.
        round_decimals : int, optional
            Coefficients that are not set to zero are rounded to the number
            of decimals specified. Defaults to 3.
    
        Returns
        -------
        sympy.core.symbol.Symbol
            The certificate in terms of correlators.
        """
        if not all([o == 2 for o in self.outcome_cardinalities]):
            raise Exception("Correlator certificates are only available " +
                            "for 2-output problems")
        try:
            cert = self.dual_certificate
        except AttributeError:
            raise Exception("For extracting a certificate you need to solve " +
                            "a problem. Call 'InflationSDP.solve()' first")
        if len(self.semiknown_moments_name_dict) > 0:  # TODO: change self.semiknown_moments_name_dict to
            Warning("Beware that, because the problem contains linearized " +
                    "polynomial constraints, the certificate is not guaranteed " +
                    "to apply to other distributions")

        names = self.names

        mons = [self.name_dict_of_monomials[name] for name in cert.keys()]
        coeffs = np.array(list(cert.values()))
        if clean and not np.allclose(coeffs, 0):
            coeffs = clean_coefficients(coeffs, chop_tol, round_decimals)
        cert = dict(zip(mons, coeffs))

        # Quick hack
        CONSTANT_MOMENT = [mon for mon in cert if mon.name == '1'][0]

        polynomial = cert[CONSTANT_MOMENT]
        for mon, coeff in cert.items():
            # First, within a factor such as P[PA_x_a, PB_y_b]
            # change the projectors, e.g. PA_x_a, to correlators
            # using PA_x_a = 1/2(1 + A_x_a).
            # Then we need to expand the product of PA_x_A * PB_y_b
            # in terms of A_x_a and B_y_b, which are treated as 
            # commuting variables, just in case we need to order
            # by party, though they should be ordered by default.
            # Afterwards, we will group together the product monomials,
            # such as  A_x_a * B_y_b, into a SINGLE commuting variable
            # with brackets, <A_x_a  B_y_b>.  Then we will expand 
            # everything and we should be done.
            if mon != CONSTANT_MOMENT:
                factors = mon.name.split('*')
                monomial = 1
                for factor in mon.factors:
                    factor = factor.tolist()
                    # Each factor is of the form "P[A_x_a, B_y_b, ..., Z_z_z]"
                    # parties, inputs, _ = np.array([t.strip().split('_') 
                    #                                for t in name[2:-1].split(',')]
                    #                               ).T.tolist()
                    parties, inputs = [], []
                    for l in factor:
                        parties.append(names[l[0] - 1])
                        inputs.append(str(l[-2]))

                    aux_prod = 1
                    for p, x in zip(parties, inputs):
                        sym = sp.Symbol(p + '_{' + x + '}', commuting=True)
                        projector = sp.Rational(1, 2) * (1 + sym)
                        aux_prod *= projector
                    aux_prod = sp.expand(aux_prod)

                    # Merge them into a single variable and add '< >' notation
                    suma = 0
                    for var, coeff1 in (sp.S.One * aux_prod).as_coefficients_dict().items():
                        if var == sp.S.One:
                            expected_value = sp.S.One
                        else:
                            # NOTE: I don't remember in which case the commented
                            # out code was useful, but I want to keep it for now
                            # if str(var)[-3:-1] == '**':
                            #     base, exp = var.as_base_exp()
                            #     if use_langlerangle:
                            #         auxname = '\langle ' + ''.join(str(base).split('*')) + ' \\rangle'
                            #     else:
                            #         auxname = '<' + ''.join(str(base).split('*')) + '>'
                            #     base = sp.Symbol(auxname, commutative=True)
                            #     expected_value = base ** exp
                            # else:
                            if use_langlerangle:
                                auxname = '\langle ' + ' '.join(str(var).split('*')) + ' \\rangle'
                            else:
                                auxname = '<' + ''.join(str(var).split('*')) + '>'
                            expected_value = sp.Symbol(auxname, commutative=True)
                        suma += coeff1 * expected_value
                    monomial *= suma
                polynomial += coeff * monomial
        polynomial = sp.expand(polynomial)

        return polynomial

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
            label = filename[:-len(extension) - 1]
        else:
            extension = 'dat-s'
            filename += '.dat-s'

        # Write file according to the extension
        if self.verbose > 0:
            print('Writing the SDP program to ', filename)
        if extension == 'dat-s':
            write_to_sdpa(self, filename)
        if extension == 'csv':
            write_to_csv(self, filename)
        elif extension == 'mat':
            write_to_mat(self, filename)

    def build_columns(self, column_specification, max_monomial_length: int = 0,
                      return_columns_numerical: bool = False) -> List[sp.core.symbol.Symbol]:
        """Process the input for the columns of the SDP relaxation.

        Parameters
        ----------
        column_specification : Union[str, List[List[int]],
                                     List[sympy.core.symbol.Symbol]]
            See description in the self.generate_relaxation()` method.
        max_monomial_length : int, optional
            Maximum number of letters in a monomial in the generating set,
            by default 0 (no limit). Example: if we choose 'local1' for 3
            parties, this gives the set {1, A, B, C, A*B, A*C, B*C, A*B*C}.
            If we set max_monomial_length=2, we remove all terms with more
            than 2 letters, and the generating set becomes:
            {1, A, B, C, A*B, A*C, B*C}.
        """
        columns = None
        if type(column_specification) == list:
            # There are two possibilities: list of lists, or list of symbols
            if type(column_specification[0]) in {list, np.ndarray}:
                if len(np.array(column_specification[1]).shape) == 2:
                    # If we are here, then the input to build columns is a list
                    # of monomials in array form, so we just return this
                    # e.g., [[0], [[1, 1, 1, 0, 0, 0]], [[1, 1, 1, 0, 0, 0],[2, 2, 1, 0, 0, 0]], ...]
                    # or [np.array([0]), np.array([[1, 1, 1, 0, 0, 0]]), np.array([[1, 1, 1, 0, 0, 0],[2, 2, 1, 0, 0, 0]]), ...]
                    columns = [np.array(mon, dtype=np.uint16) for mon in column_specification]
                elif len(np.array(column_specification[1]).shape) == 1:
                    # If the depth of column_specification is just 2,
                    # then the input must be in the form of
                    # e.g., [[], [0], [1], [0, 0]] -> {1, A{:}, B{:}, (A*A){:}}
                    # which just specifies the party structure in the
                    # generating set
                    columns = self._build_cols_from_col_specs(column_specification)
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
                            columns += [np.array([[0]], dtype=np.uint16)]
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
                max_length = max_monomial_length if max_monomial_length > 0 and max_monomial_length < npa_level else npa_level
                for length in range(1, max_length + 1):
                    for number_tuple in itertools.product(*[range(self.nr_parties) for _ in range(length)]):
                        a = np.array(number_tuple)
                        if np.all(a[:-1] <= a[1:]):
                            # if tuple in increasing order from left to right
                            col_specs += [a.tolist()]
                columns = self._build_cols_from_col_specs(col_specs)


            elif 'local' in column_specification.lower():
                local_level = int(column_specification[5:])
                local_length = local_level * self.nr_parties  # max 1 operators per party
                max_length = max_monomial_length if max_monomial_length > 0 and max_monomial_length < local_length else local_length

                party_frequencies = []
                for pfreq in itertools.product(*[range(local_level + 1) for _ in range(self.nr_parties)]):
                    if sum(pfreq) <= max_length:
                        party_frequencies.append(list(reversed(pfreq)))
                party_frequencies = sorted(party_frequencies, key=sum)

                col_specs = []
                for pfreq in party_frequencies:
                    lst = []
                    for party in range(self.nr_parties):
                        lst += [party] * pfreq[party]
                    col_specs += [lst]
                columns = self._build_cols_from_col_specs(col_specs)

            elif 'physical' in column_specification.lower():
                try:
                    inf_level = int(column_specification[8])
                    length = len(column_specification[8:])
                    message = ("Physical monomial generating set party number" +
                               "specification must have length equal to 1 or " +
                               "number of parties. E.g.: For 3 parties, " +
                               "'physical322'.")
                    assert length == self.nr_parties or length == 1, message
                    if length == 1:
                        physmon_lens = [inf_level] * self.nr_sources
                    else:
                        physmon_lens = [int(inf_level)
                                        for inf_level in column_specification[8:]]
                    max_total_mon_length = sum(physmon_lens)
                except:
                    # If no numbers come after, by default we use all physical operators
                    physmon_lens = self.inflation_levels
                    max_total_mon_length = sum(physmon_lens)

                if max_monomial_length > 0:
                    max_total_mon_length = max_monomial_length

                party_frequencies = []
                for pfreq in itertools.product(*[range(physmon_lens[party] + 1)
                                                 for party in range(self.nr_parties)]):
                    if sum(pfreq) <= max_total_mon_length:
                        party_frequencies.append(list(reversed(pfreq)))
                party_frequencies = sorted(party_frequencies, key=sum)

                physical_monomials = []
                for freqs in party_frequencies:
                    if freqs == [0] * self.nr_parties:
                        physical_monomials.append(np.array([[0]], dtype=np.uint16))
                    else:
                        physmons_per_party_per_length = []
                        for party, freq in enumerate(freqs):
                            # E.g., if freq = [1, 2, 0], then
                            # physmons_per_party_per_length will be a list
                            # of lists of physical monomials of length 1, 2,
                            # and 0 physmons_per_party_per_length =
                            # [template_of_len_1_party0, template_of_len_2_party1]
                            # and nothing for party 'C' since we have a 0 frequency
                            if freq > 0:
                                # template_of_len_party_0 = template_physmon_all_lens[freq-1]
                                # with_correct_party_idx = physmon_change_party_in_template(template_of_len_party_0, party)
                                physmons = phys_mon_1_party_of_given_len(self.hypergraph,
                                                                         self.inflation_levels,
                                                                         party, freq,
                                                                         self.setting_cardinalities,
                                                                         self.outcome_cardinalities,
                                                                         self.names,
                                                                         self._lexorder)
                                physmons_per_party_per_length.append(physmons)

                        for mon_tuple in itertools.product(*physmons_per_party_per_length):
                            physical_monomials.append(to_canonical(np.concatenate(mon_tuple),
                                                                   self._notcomm,
                                                                   self._lexorder))

                columns = physical_monomials
            else:
                raise Exception('I have not understood the format of the '
                                + 'column specification')
        else:
            raise Exception('I have not understood the format of the '
                            + 'column specification')

        # Sort by custom lex order?
        if not np.array_equal(self._lexorder, self._default_lexorder):
            res_lexrepr = [nb_mon_to_lexrepr(m, self._lexorder).tolist()
                           if m.tolist() != [[0]] else []
                           for m in columns]
            sortd = sorted(res_lexrepr, key=lambda x: (len(x), x))
            columns = [self._lexorder[lexrepr]
                       if lexrepr != [] else np.array([[0]], dtype=np.uint16)
                       for lexrepr in sortd]

        columns_symbolical = [to_symbol(col, self.names) for col in columns]

        if return_columns_numerical:
            return columns_symbolical, columns
        else:
            return columns_symbolical

    def _build_cols_from_col_specs(self, col_specs: List[List]) -> List[np.ndarray]:
        """This builds the generating set for the moment matrix taking as input
        a block specifying only the number of parties, and the party labels.

        For example, with col_specs=[[], [0], [2], [0, 2]] as input, we
        generate the generating set S={1, A_{ijk}_xa, C_{lmn}_zc,
        A_{i'j'k'}_x'a' * C{l'm'n'}_{z'c'}} where i,j,k,l,m,n,i',j',k',l',m',n'
        represent all possible inflation copies indices compatible with the
        network structure, and x,a,z,c,x',a',z',c' are all possible input
        and output indices compatible with the cardinalities. 
        
        As further examples, NPA level 2 for 3 parties is built from
        [[], [0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
        and "local level 1" is built from
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

        # if self.substitutions == {}:
        #     if all([len(block) == len(np.unique(block)) for block in col_specs]):
        #         warn("You have not input substitution rules to the " +
        #              "generation of columns, but it is OK because you " +
        #              "are using local level 1")
        #     else:
        #         raise Exception("You must input substitution rules for columns "
        #                         + "to be generated properly")
        # else:

        res = []
        allvars = set()
        for block in col_specs:
            if block == []:
                res.append(np.array([[0]], dtype=np.uint16))
                allvars.add('1')
            else:
                meas_ops = []
                for party in block:
                    meas_ops.append(flatten(self.measurements[party]))
                for monomial_factors in itertools.product(*meas_ops):
                    mon = np.array([to_numbers(op, self.names)[0]
                                    for op in monomial_factors], dtype=np.uint16)
                    if self.commuting:
                        canon = remove_projector_squares(mon_lexsorted(mon, self._lexorder))
                        if mon_is_zero(canon):
                            canon = 0
                    else:
                        canon = to_canonical(mon, self._notcomm, self._lexorder)
                    if not np.array_equal(canon, 0):
                        # If the block is [0, 0], and we have the monomial
                        # A**2 which simplifies to A, then A would be included
                        # in the block [0]. This is not a problem, but we use
                        # the convention that [0, 0] means all monomials of
                        # length 2 AFTER simplifications, so we omit monomials
                        # of length 1.
                        if canon.shape[0] == len(monomial_factors):
                            name = to_name(canon, self.names)
                            if name not in allvars:
                                allvars.add(name)
                                if name == '1':
                                    # TODO: Convention in this branch is to never use to_name or to_numbers. Hashing done via tuple format.
                                    res.append(np.array([[0]], dtype=np.uint16))
                                else:
                                    res.append(canon)

        return res

    def _generate_parties(self):  # TODO Remove all traces of ncpol2sdpa
        # TODO: change name to generate_measurements
        """Generates all the party operators and substitution rules in an
        quantum inflation setup on a network given by a hypergraph. It uses
        information stored in the `InflationSDP` instance.

        It stores in `self.measurements` a list of lists of measurement
        operators indexed as self.measurements[p][c][i][o] for party p,
        copies c, input i, output o.

        It stores in `self.substitutions` a dictionary of substitution rules
        containing commutation, orthogonality and square constraints. NOTE:
        to use these substitutions in the moment matrix we need to use
        ncpol2sdpa.
        """

        hypergraph = self.hypergraph
        settings = self.setting_cardinalities
        outcomes = self.outcome_cardinalities
        inflation_level_per_source = self.inflation_levels
        commuting = self.commuting

        assert len(settings) == len(
            outcomes), 'There\'s a different number of settings and outcomes'
        assert len(
            settings) == hypergraph.shape[1], 'The hypergraph does not have as many columns as parties'

        substitutions = {}
        measurements = []
        parties = self.names
        n_states = hypergraph.shape[0]
        for pos, [party, ins, outs] in enumerate(zip(parties, settings, outcomes)):
            party_meas = []
            # Generate all possible copy indices for a party
            # party_states = sum(hypergraph[:, pos])
            # all_inflation_indices = itertools.product(range(inflation_level), repeat=party_states)
            all_inflation_indices = itertools.product(
                *[list(range(inflation_level_per_source[p_idx]))
                  for p_idx in np.nonzero(hypergraph[:, pos])[0]])
            # Include zeros in the positions corresponding to states not feeding the party
            all_indices = []
            for inflation_indices in all_inflation_indices:
                indices = []
                i = 0
                for idx in range(n_states):
                    if hypergraph[idx, pos] == 0:
                        indices.append('0')
                    elif hypergraph[idx, pos] == 1:
                        # The +1 is just to begin at 1
                        indices.append(str(inflation_indices[i] + 1))
                        i += 1
                    else:
                        raise Exception('You don\'t have a proper hypergraph')
                all_indices.append(indices)

            # Generate measurements for every combination of indices.
            for indices in all_indices:
                meas = generate_noncommuting_measurements([outs - 1 for _ in range(ins)],
                                                          # outs -1 because of CG notation
                                                          party + '_' + '_'.join(indices))

                party_meas.append(meas)
            measurements.append(party_meas)

        substitutions = {}

        if commuting:
            flatmeas = flatten(measurements)
            for i in range(len(flatmeas)):
                for j in range(i, len(flatmeas)):
                    m1 = flatmeas[i]
                    m2 = flatmeas[j]
                    if str(m1) > str(m2):
                        substitutions[m1 * m2] = m2 * m1
                    elif str(m1) < str(m2):
                        substitutions[m2 * m1] = m1 * m2
                    else:
                        pass
        else:
            # Commutation of different parties
            for i in range(len(parties)):
                for j in range(len(parties)):
                    if i > j:
                        for op1 in flatten(measurements[i]):
                            for op2 in flatten(measurements[j]):
                                substitutions[op1 * op2] = op2 * op1
            # Operators for a same party with non-overlapping copy indices commute
            for party, inf_measurements in enumerate(measurements):
                sources = hypergraph[:, party].astype(bool)
                inflation_indices = [np.compress(sources,
                                                 str(flatten(inf_copy)[0]).split('_')[1:-2]).astype(int)
                                     for inf_copy in inf_measurements]
                for ii, first_copy in enumerate(inf_measurements):
                    for jj, second_copy in enumerate(inf_measurements):
                        if (jj > ii) and (all(inflation_indices[ii] != inflation_indices[jj])):
                            for op1, op2 in itertools.product(flatten(first_copy),
                                                              flatten(second_copy)):
                                substitutions[op2 * op1] = op1 * op2

        for party in measurements:
            # Idempotency
            substitutions = {**substitutions,
                             **{op ** 2: op for op in flatten(party)}}
            # Orthogonality
            for inf_copy in party:
                for measurement in inf_copy:
                    for out1 in measurement:
                        for out2 in measurement:
                            if out1 == out2:
                                substitutions[out1 * out2] = out1
                            else:
                                substitutions[out1 * out2] = 0

        return measurements, substitutions, parties

    def _build_momentmatrix(self) -> Tuple[np.ndarray, Dict]:
        """Generate the moment matrix.
        """

        _cols = [np.array(col, dtype=np.uint16)
                 for col in self.generating_monomials]
        problem_arr, canonical_mon_to_idx_dict = calculate_momentmatrix(_cols,
                                                                        self._notcomm,
                                                                        self._lexorder,
                                                                        verbose=self.verbose,
                                                                        commuting=self.commuting)
        idx_to_canonical_mon_dict = {idx: np.asarray(mon, dtype=np.uint16) for (mon, idx) in canonical_mon_to_idx_dict.items() if idx >= 2}

        return problem_arr, idx_to_canonical_mon_dict

    def _calculate_inflation_symmetries(self) -> np.ndarray:
        """Calculates all the symmetries and applies them to the set of
        operators used to define the moment matrix. The new set of operators
        is a permutation of the old. The function outputs a list of all
        permutations.

        Returns
        -------
        List[List[int]]
            Returns a list of all permutations of the operators implied by
            the inflation symmetries.
        """

        inflevel = self.inflation_levels
        n_sources = self.nr_sources

        inflation_symmetries = []  # [list(range(len(self.generating_monomials)))]

        # # TODO do this function without relying on symbolic substitutions!!
        # flatmeas  = np.array(flatten(self.measurements))
        # measnames = np.array([str(meas) for meas in flatmeas])

        list_original = from_numbers_to_flat_tuples(self.generating_monomials)
        for source, permutation in tqdm(sorted(
                [(source, permutation) for source in list(range(n_sources))
                 for permutation in itertools.permutations(range(inflevel[source]))]
        ),
                disable=not self.verbose,
                desc="Calculating symmetries       "):
            permuted_cols_ind = \
                apply_source_permutation_coord_input(self.generating_monomials,
                                                     source,
                                                     permutation,
                                                     self.commuting,
                                                     self._notcomm,
                                                     self._lexorder)
            # self.substitutions,
            # flatmeas,
            # measnames,
            # self.names)
            # Bring columns to the canonical form using commutations of
            # connected components
            for idx in range(len(permuted_cols_ind)):
                if len(permuted_cols_ind[idx].shape) > 1:
                    permuted_cols_ind[idx] = to_canonical(permuted_cols_ind[idx],
                                                          self._notcomm,
                                                          self._lexorder)
                    # permuted_cols_ind[idx] = np.vstack(
                    #              sorted(np.concatenate(factorize_monomial(
                    #                                       permuted_cols_ind[idx]
                    #                                                       )),
                    #                     key=lambda x: x[0])
                    #                                    )
            list_permuted = from_numbers_to_flat_tuples(permuted_cols_ind)
            try:
                total_perm = find_permutation(list_permuted, list_original)
                inflation_symmetries.append(total_perm)
            except:
                if self.verbose > 0:
                    warnings.warn("The generating set is not closed under source swaps." +
                                  "Some symmetries will not be implemented.")

        return np.unique(inflation_symmetries, axis=0)

    # @staticmethod
    # def _symmetrize_moment_matrix_via_sympy(momentmatrix: np.ndarray,
    #                               inflation_symmetries: np.ndarray):
    #     initial_group = np.asarray(dimino_sympy(inflation_symmetries))
    #     template_matrix = np.arange(np.prod(momentmatrix.shape)).reshape()
    #     elevated_group = []
    #     for perm in initial_group:
    #         permuted_template = matrix_permute(template_matrix, perm)
    #         elevated_group.append(permuted_template.ravel())
    #         elevated_group.append(permuted_template.T.ravel())
    #     return elevated_group

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


        Returns
        -------
        Tuple[np.ndarray, Dict[int, int], np.ndarray]
            The symmetrized moment matrix, the orbits as a dictionary
            and the symmetrized monomials list.
        """

        # if len(symmetric_arr.shape) == 2:
        #     # TODO This is inelegant, remove the index.
        #     #  Only here for compatibility reasons
        #     aux = np.zeros((symmetric_arr.shape[0], symmetric_arr.shape[1], 2))
        #     aux[:, :, 0] = symmetric_arr
        #     symmetric_arr = aux.astype(int)

        # indices_to_delete = []
        # the +2 is to include 0:0 and 1:1
        # orbits = {i: i for i in range(2+len(monomials_list))}
        # orbits = {i: i for i in np.unique(sdp.problem_arr.flat)}
        orbits, where_it_matters_flat = np.unique(momentmatrix.flat, return_index=True)
        # (where_it_matters_rows, where_it_matters_cols) = np.unravel_index(where_it_matters_flat, momentmatrix.shape)
        # orbits = np.unique(momentmatrix.data)
        absent_indices = np.arange(np.min(orbits))
        orbits = np.concatenate((absent_indices, orbits))
        # print("orbits before symmetrization", orbits)
        symmetric_arr = momentmatrix.copy()

        for permutation in tqdm(inflation_symmetries,
                                disable=not self.verbose,
                                desc="Applying symmetries          "):
            if conserve_memory:
                for i, ip in enumerate(permutation):
                    for j, jp in enumerate(permutation):
                        if symmetric_arr[i, j] < symmetric_arr[ip, jp]:
                            # indices_to_delete.append(int(symmetric_arr[ip, jp]))
                            # orbits[symmetric_arr[ip, jp]] = symmetric_arr[i, j]
                            symmetric_arr[ip, jp] = symmetric_arr[i, j]
            else:
                np.minimum(symmetric_arr, (symmetric_arr[permutation].T)[permutation].T, out=symmetric_arr)
        orbits = np.concatenate((absent_indices, symmetric_arr.flat[where_it_matters_flat].flat))

        # print("orbits before adjustment", orbits)

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
        # print("orbits after adjustment", orbits)

        old_representative_indices, new_indices, unsym_idx_to_sym_idx = np.unique(orbits,
                                                                                  return_index=True,
                                                                                  return_inverse=True)
        assert np.array_equal(old_representative_indices, new_indices
                              ), 'Something unexpected happened when calculating orbits.'

        symmetric_arr = unsym_idx_to_sym_idx.take(momentmatrix)
        symidx_to_canonical_mon_dict = {new_idx: unsymidx_to_canonical_mon_dict[old_idx] for new_idx, old_idx in
                                        enumerate(
                                            old_representative_indices) if old_idx >= 2}

        # return symmetric_arr, orbits, symidx_to_canonical_mon_dict
        return symmetric_arr, unsym_idx_to_sym_idx, symidx_to_canonical_mon_dict

    def dump_to_file(self, filename):
        """
        Save the whole object to a file using `pickle`.

        Parameters
        ----------
        filename : str
            Name of the file.
        """
        with open(filename, 'w') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def monomial_sanity_check(self, monomial):
        """
        The hypergraph corresponding to the monomial should match a
        subgraph of the scenario hypergraph.
        """
        parties = np.asarray(monomial)[:, 0].astype(int)
        # Parties start at #1 in our numpy vector notation.
        scenario_subhypergraph = self.hypergraph[:, parties - 1]
        monomial_sources = np.asarray(monomial)[:, 1:-2].T
        monomial_hypergraph = monomial_sources.copy()
        monomial_hypergraph[np.nonzero(monomial_hypergraph)] = 1
        return all([source in scenario_subhypergraph.tolist()
                    for source in monomial_hypergraph.tolist()])


if __name__ == "__main__":
    # sdp = InflationSDP(InflationProblem(dag={'U_AB': ['A','B'],
    #                                    'U_AC': ['A','C'],
    #                                    'U_AD': ['A','D'],
    #                                    'C': ['D'],
    #                                    'A': ['B', 'C', 'D']},
    #                               outcomes_per_party=(2, 2, 2, 2),
    #                               settings_per_party=(1, 1, 1, 1),
    #                               inflation_level_per_source=(1, 1, 1),
    #                               names=('A', 'B', 'C', 'D'),
    #                               verbose=2),
    #                    commuting=False,
    #                    verbose=2)
    # sdp.generate_relaxation('local1')

    cutInflation = InflationProblem({"lambda": ["a", "b"],
                                     "mu": ["b", "c"],
                                     "sigma": ["a", "c"]},
                                    order=['a', 'b', 'c'],
                                    outcomes_per_party=[2, 2, 2],
                                    settings_per_party=[1, 1, 1],
                                    inflation_level_per_source=[2, 1, 1])
    sdp = InflationSDP(cutInflation)
    sdp.generate_relaxation('local1')

    # print(sdp.unsymmetrized_mm_idxs)
    # print(list(sdp.unsymidx_to_canonical_mon_dict.items()))
    # print(sdp.momentmatrix)
    # print(len(sdp.symidx_to_canonical_mon_dict))
    # print(len(sdp.symidx_to_Monomials_dict))
    #
    # for k, mon in sdp.symidx_to_Monomials_dict.items():
    #
    #     print(f"{k} := {mon}, {mon.knowability_status}")
