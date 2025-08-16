"""
The module generates the semidefinite program associated to a quantum inflation
instance (see arXiv:1909.10519).

@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""
from collections import Counter, deque, defaultdict
from functools import reduce, cached_property
from gc import collect
from itertools import chain, count, product, repeat, combinations
from operator import neg as op_neg, itemgetter
from numbers import Real
from typing import List, Dict, Tuple, Union, Any
from warnings import warn

import numpy as np
import sympy as sp
from scipy.sparse import coo_array
from tqdm import tqdm

from .. import InflationProblem
from .fast_npa import nb_is_knowable as is_knowable
from .fast_npa import (reverse_mon,
                       to_canonical_1d_internal
                       )
from .monomial_classes import InternalAtomicMonomialSDP, CompoundMomentSDP
from .quantum_tools import (apply_inflation_symmetries,
                            calculate_momentmatrix_1d_internal,
                            construct_normalization_eqs,
                            flatten_symbolic_powers
                            )
from .sdp_utils import solveSDP_MosekFUSION
from .writer_utils import (write_to_csv,
                           write_to_mat,
                           write_to_sdpa)
from ..lp.numbafied import nb_outer_bitwise_or
from ..utils import clean_coefficients, partsextractor, eprint


class InflationSDP:
    """Class for generating and solving an SDP relaxation for quantum inflation.
    """
    constant_term_name = "constant_term"

    def __init__(self,
                 inflationproblem: InflationProblem,
                 supports_problem: bool = False,
                 include_all_outcomes: bool = False,
                 commuting: bool = False,
                 real_qt: bool = False,
                 verbose: int = None) -> None:
        """
        Class for generating and solving an SDP relaxation for quantum inflation.

        Parameters
        ----------
        inflationproblem : InflationProblem
            Details of the scenario.
        supports_problem : bool, optional
            Whether to consider feasibility problems with distributions, or just
            with the distribution's support. By default ``False``.
        real_qt : bool, optional
            Whether to assume real quantum theory instead of traditional (complex)
            quantum theory. By default ``False``.
        verbose : int, optional
            Optional parameter for level of verbose:

                * 0: quiet (default),
                * 1: monitor level: track program process and show warnings,
                * 2: debug level: show properties of objects created.
        """
        
        self.problem_type = "sdp"
        self.supports_problem = supports_problem
        if verbose is not None:
            if inflationproblem.verbose > verbose:
                warn("Overriding the verbosity from InflationProblem")
            self.verbose = verbose
        else:
            self.verbose = inflationproblem.verbose
        self.InflationProblem = inflationproblem
        self.names = inflationproblem.names
        self.names_to_ints = {name: i + 1 for i, name in enumerate(self.names)}
        if self.verbose > 1:
            eprint(inflationproblem)

        self.nr_parties = len(self.names)
        self.nr_sources = inflationproblem.nr_sources
        self.hypergraph = inflationproblem.hypergraph
        self.inflation_levels = inflationproblem.inflation_level_per_source
        self.has_children = inflationproblem.has_children
        self.outcome_cardinalities = inflationproblem.outcomes_per_party.copy()
        self.has_children = inflationproblem.has_children.copy()
        if include_all_outcomes or supports_problem:  
            # HACK to fix detection of incompatible supports. 
            # (Can be fixed upon adding set_extra_equalities)
            self.has_children[:] = True
        self.does_not_have_children = np.logical_not(self.has_children)


        self.outcome_cardinalities += self.has_children
        self.setting_cardinalities = inflationproblem.settings_per_party
        self._quantum_sources = inflationproblem._nonclassical_sources

        if self.verbose > 1:
            eprint("Number of single operator measurements per party:", end="")
            prefix = " "
            for i, measures in enumerate(self.InflationProblem.measurements_symbolic):
                counter = count()
                deque(zip(chain.from_iterable(
                    chain.from_iterable(measures)),
                          counter),
                      maxlen=0)
                eprint(prefix + f"{self.names[i]}={next(counter)}", end="")
                prefix = ", "
            eprint()
        self.use_lpi_constraints = False
        self.network_scenario    = inflationproblem.is_network
        self._is_knowable_q_non_networks = \
            inflationproblem._is_knowable_q_non_networks
        self.rectify_fake_setting = inflationproblem.rectify_fake_setting
        self.factorize_moment_1d = inflationproblem.factorize_monomial_1d

        self._nr_properties = inflationproblem._nr_properties
        self.np_dtype = inflationproblem._np_dtype
        self._astuples_dtype = inflationproblem._astuples_dtype
        self.identity_operator = np.empty((0, self._nr_properties),
                                          dtype=self.np_dtype)
        self.zero_operator = np.zeros((1, self._nr_properties),
                                      dtype=self.np_dtype)

        self._default_lexorder = np.concatenate((self.zero_operator, 
                                                 inflationproblem._lexorder)
                                                )
        self._nr_operators = inflationproblem._nr_operators + 1
        self.blank_bool_array = np.zeros((1, self._nr_operators), dtype=bool)
        self._lexeye = np.eye(self._nr_operators, dtype=bool)
        all_ortho_groups_as_boolarrays = []
        for ortho_groups in inflationproblem._ortho_idxs_per_party:
            ortho_groups_as_boolarrays = []
            for ortho_group in ortho_groups:
                ortho_groups_as_boolarrays.append(self._lexeye[ortho_group])
            all_ortho_groups_as_boolarrays.append(ortho_groups_as_boolarrays)
        self._CG_limited_ortho_groups_as_boolarrays = []
        self._raw_CG_ops = []
        for i, ortho_groups_as_boolarrays in enumerate(all_ortho_groups_as_boolarrays):
            if self.does_not_have_children[i]:
                for ortho_group_as_boolarray in ortho_groups_as_boolarrays:
                    self._CG_limited_ortho_groups_as_boolarrays.append(ortho_group_as_boolarray[:-1])
                    self._raw_CG_ops.extend(np.flatnonzero(ortho_group_as_boolarray[-1]))
            else:
                self._CG_limited_ortho_groups_as_boolarrays.extend(ortho_groups_as_boolarrays)
        self._lexorder = self._default_lexorder.copy()
        self.op_to_lexrepr_dict = {tuple(op): i for i, op in enumerate(self._lexorder)}
        self._lexorder_len = len(self._lexorder)
        self.raw_lexorder_symmetries = \
            np.pad(inflationproblem.symmetries + 1, ((0, 0), (1, 0)))
        self._CG_ops = np.sort(self._raw_CG_ops).astype(int)+1
        self.lexorder_symmetries=np.array([
            perm for perm in self.raw_lexorder_symmetries
            if np.array_equal(np.sort(perm[self._CG_ops]), self._CG_ops)
        ], dtype=int)
        if self.verbose > 0:
            old_group_size = len(self.raw_lexorder_symmetries)
            new_group_size = len(self.lexorder_symmetries)
            if new_group_size < old_group_size:
                eprint("Warning: The use of Collins-Gisin notation internally via the argument `include_all_outcomes=False`")
                eprint(f" means that not all symmetries of the problem can be exploited. Group size drop from {old_group_size} to {new_group_size}.")

        self._lexrepr_to_names = \
            np.hstack((["0"], inflationproblem._lexrepr_to_names)).astype(object)
        self._lexrepr_to_copy_index_free_names = \
            np.hstack((["0"], inflationproblem._lexrepr_to_copy_index_free_names)).astype(object)
        self.op_from_name = {"0": 0}
        for i, op_names in enumerate(inflationproblem._lexrepr_to_all_names.tolist()):
            for op_name in op_names:
                self.op_from_name.setdefault(op_name, i+1)
        self._lexrepr_to_symbols = \
            np.hstack(([sp.S.Zero], inflationproblem._lexrepr_to_symbols))

        #Construct orthogonality matrix for recognizing zeros
        self._orthomat = np.zeros((self._lexorder_len, self._lexorder_len),
                                  dtype=bool)
        for ((i, j), (op_i, op_j)) in zip(
                combinations(range(self._lexorder_len), 2),
                combinations(self._lexorder, 2)):
            if (op_i[-1] != op_j[-1] and np.array_equal(op_i[:-1], op_j[:-1])):
                self._orthomat[i, j] = True
                self._orthomat[j, i] = True
        self._orthomat[:, 0] = True
        self._orthomat[0, :] = True

        # Translating the compatibility matrix of InflationProblem to
        # a commutativity matrix for InflationSDP.
        # InflationProblem has more operators in ._lexorder than InflationSDP
        # This is because events with the last outcome are included in
        # InflationProblem. We carefully avoid this by using .mon_to_lexrepr
        # of InflationProblem on the operators in InflationSDP._lexorder
        assert np.allclose(self._lexorder[0], self.zero_operator), \
            "The first element of the lexorder should be the zero operator"
        self._default_notcomm = \
            np.pad(inflationproblem._default_notcomm,
                       ((1, 0), (1, 0)))

        self._notcomm = self._default_notcomm.copy()
        self.all_operators_commute = not self._notcomm.any()
        if commuting:
            assert self.all_operators_commute, \
                "You appear to be requesting commuting (classical)" \
                    + " inflation, \nbut have not specified classical_sources=`all`." \
                    + "\nNote that the `commuting` keyword argument has been deprecated as of release 2.0.0"
        if real_qt:
            warn("Support for real quantum theory is experimental. " + 
                 "Use at your own risk.")
            assert not self.all_operators_commute, \
                "You appear to be requesting inflation assuming real quantum theory," \
                + " but this is meaningless without noncommuting operators."
            self.real_qt = True
        else:
            self.real_qt = False
        if self.all_operators_commute:
            self.all_commuting_q_2d = lambda mon: True
            self.all_commuting_q_1d = lambda lexmon: True
        else:
            self.all_commuting_q_1d = \
                lambda lexmon: not self._notcomm[np.ix_(lexmon, lexmon)].any()
            self.all_commuting_q_2d = \
                lambda mon: self.all_commuting_q_1d(self.mon_to_lexrepr(mon))

        self.canon_lexmon_from_hash     = {}
        self.canonsym_lexmon_from_hash  = {}
        # These next properties are reset during generate_relaxation, but
        # are needed in init so as to be able to test the Monomial constructor
        # function without generate_relaxation.
        self.atomic_monomial_from_hash  = {}
        self.monomial_from_atoms        = {}
        self.monomial_from_name         = {}
        self.monomial_from_symbol       = {}
        self.Zero = self.Moment_2d(self.zero_operator, idx=0)
        self.One  = self.Moment_2d(self.identity_operator, idx=1)
        self._relaxation_has_been_generated = False
        self.expected_distro_shape = inflationproblem.expected_distro_shape

    ###########################################################################
    # MAIN ROUTINES EXPOSED TO THE USER                                       #
    ###########################################################################
    def generate_relaxation(self,
                            column_specification:
                            Union[str,
                                  List[List[int]],
                                  List[sp.core.symbol.Symbol]] = "npa1",
                            **kwargs
                            ) -> None:
        r"""Creates the SDP relaxation of the quantum inflation problem using
        the `NPA hierarchy <https://www.arxiv.org/abs/quant-ph/0607119>`_ and
        applies the symmetries inferred from inflation.

        It takes as input the generating set of monomials :math:`\{M_i\}_i`.
        The moment matrix :math:`\Gamma` is defined by all the possible inner
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

            * `(str)` ``"npaN"``: where N is an integer. This represents level
              N in the Navascues-Pironio-Acin hierarchy
              (`arXiv:quant-ph/0607119
              <https://www.arxiv.org/abs/quant-ph/0607119>`_).
              For example, level 3 with measurements :math:`\{A, B\}` will give
              the set :math:`\{1, A, B, AA, AB, BB, AAA, AAB, ABB, BBB\}` for
              all inflation, input and output indices. This hierarchy is known
              to converge to the quantum set for :math:`N\rightarrow\infty`.

            * `(str)` ``"localN"``: where N is an integer. Local level N
              considers monomials that have at most N measurement operators per
              party. For example, ``local1`` is a subset of ``npa2``; for two
              parties, ``npa2`` is :math:`\{1, A, B, AA, AB, BB\}` while
              ``local1`` is :math:`\{1, A, B, AB\}`.

            * `(str)` ``"physicalN"``: The subset of local level N with only
              operators that have non-negative expectation values with any
              state. N cannot be greater than the smallest number of copies of
              a source in the inflated graph. For example, in the scenario
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
              [1], [2], [0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]`` is
              the same as :math:`\{1, A, B, C, AA, AB, AC, BB, BC, CC\}`, which
              is the same as ``npa2`` for three parties. ``[[]]`` encodes the
              identity element.

            * `List[sympy.core.symbol.Symbol]`: one can also fully specify the
              generating set by giving a list of symbolic operators built from
              the measurement operators in ``InflationSDP.measurements``. This
              list needs to have the identity ``sympy.S.One`` as the first
              element.

        kwargs :
            Additional arguments that will be passed to
            ``InflationSDP.build_columns``.
        """
        self.atomic_monomial_from_hash  = {}
        self.monomial_from_atoms        = {}
        self.monomial_from_name         = {}
        self.monomial_from_symbol       = {}
        self.Zero = self.Moment_2d(self.zero_operator, idx=0)
        self.One  = self.Moment_2d(self.identity_operator, idx=1)
        self.Constant_Term = self.One.__copy__()
        self.Constant_Term.name = self.constant_term_name
        self.monomial_from_name[self.constant_term_name] = self.Constant_Term

        self.build_columns(column_specification, **kwargs)
        collect()
        if self.verbose > 0:
            eprint("Number of columns in the moment matrix:", self.n_columns)

        # Calculate the moment matrix without the inflation symmetries
        unsymmetrized_mm, unsymmetrized_corresp = \
            self._build_momentmatrix_1d_internal()
        symmetrization_required = np.any(self.inflation_levels - 1)
        additional_var = 0
        if self.verbose > 1:
            extra_msg = (" before symmetrization" if symmetrization_required
                         else "")
            if 0 in unsymmetrized_mm.flat:
                additional_var = 1
            eprint("Number of variables" + extra_msg + ":",
                  len(unsymmetrized_corresp) + additional_var)

        # Calculate the inflation symmetries
        self.columns_symmetries = self._discover_columns_symmetries()

        # Apply the inflation symmetries to the moment matrix
        self.momentmatrix, self.orbits, representative_unsym_idxs = \
            apply_inflation_symmetries(unsymmetrized_mm,
                                       self.columns_symmetries,
                                       self.verbose)
        self.symmetrized_corresp = \
            {self.orbits[idx]: unsymmetrized_corresp[idx]
             for idx in representative_unsym_idxs.flat if idx >= 1}
        if self.verbose > 0:
            extra_msg = (" after symmetrization" if symmetrization_required
                         else "")
            eprint(f"Number of variables{extra_msg}: "
                  + f"{len(self.symmetrized_corresp)+additional_var}")
        del unsymmetrized_mm, unsymmetrized_corresp, \
            symmetrization_required, additional_var
        collect()

        self.momentmatrix_has_a_zero, self.momentmatrix_has_a_one = \
            np.isin([0, 1], self.momentmatrix.ravel())

        # Associate Monomials to the remaining entries. The zero monomial is
        # not stored during calculate_momentmatrix
        first_free_index = 0
        self.compmoment_from_idx = dict()
        _compmonomial_to_idx = dict()
        self.n_vars = len(self.symmetrized_corresp)
        _compmonomial_to_idx[self.Zero] = 0
        self.compmoment_from_idx[0] = self.Zero
        first_free_index += 1
        self.n_vars += 1
        self.extra_inverse = np.arange(self.n_vars, dtype=int)
        self.old_indices_associated_with_new_index = defaultdict(list)
        self.old_indices_associated_with_monomial = defaultdict(list)
        self.old_indices_associated_with_new_index[0]=[0]
        self.old_indices_associated_with_monomial[self.Zero] = [0]
        for (idx, lexmon) in tqdm(self.symmetrized_corresp.items(),
                             disable=not self.verbose,
                             desc="Initializing monomials   ",
                             total=self.n_vars):
            mon = self.Moment_1d(lexmon, first_free_index)
            self.compmoment_from_idx[idx] = mon # Critical for normalization equations and other functions that use old indices
            try:
                current_index = _compmonomial_to_idx[mon]
                mon.idx = current_index
            except KeyError:
                current_index = first_free_index
                _compmonomial_to_idx[mon] = current_index
                first_free_index += 1
            self.old_indices_associated_with_new_index[current_index].append(idx)
            self.old_indices_associated_with_monomial[mon].append(idx)
            self.extra_inverse[idx] = current_index

        self.monomials = list(_compmonomial_to_idx.keys())
        if not self.momentmatrix_has_a_zero:
            self.monomials = self.monomials[1:]
            self.n_vars -= 1
        self.moments = self.monomials
        old_num_vars = self.n_vars
        self.n_vars = len(self.monomials)
        self.first_free_idx = first_free_index
        if self.n_vars < old_num_vars:
            if self.verbose > 0:
                eprint("Further variable reduction has been made possible. Number of variables in the SDP:",
                       self.n_vars)
        del _compmonomial_to_idx
        collect(generation=2)

        _counter = Counter([mon.knowability_status for mon in self.moments])
        self.n_knowable           = _counter["Knowable"]
        self.n_something_knowable = _counter["Semi"]
        self.n_unknowable         = _counter["Unknowable"]
        if self.verbose > 1:
            eprint(f"The problem has {self.n_knowable} knowable moments, " +
                  f"{self.n_something_knowable} semi-knowable moments, " +
                  f"and {self.n_unknowable} unknowable moments.")

        if self.all_operators_commute:
            self.hermitian_moments = self.moments
            self.physical_moments = self.moments
        else:
            self.hermitian_moments = [mon for mon in self.moments
                                      if mon.is_hermitian]
            self.physical_moments = [mon for mon in self.hermitian_moments
                                     if mon.is_all_commuting]
            if self.verbose > 1:
                eprint(f"The problem has {len(self.hermitian_moments)} " +
                      "non-negative moments.")

        # This dictionary useful for certificates_as_probs
        self.names_to_symbols = {mon.name: mon.symbol
                                 for mon in self.moments}
        self.names_to_symbols[self.constant_term_name] = sp.S.One

        # In non-network scenarios we do not use Collins-Gisin notation for
        # some variables, so there exist normalization constraints between them
        self.minimal_equalities = []
        if not self.network_scenario or self.supports_problem:
            self.column_level_equalities = self._discover_normalization_eqns()
            self.idx_level_equalities    = construct_normalization_eqs(
                                                self.column_level_equalities,
                                                self.momentmatrix,
                                                self.verbose)
            if self.verbose > 1 and len(self.idx_level_equalities):
                eprint("Number of normalization equalities:",
                      len(self.idx_level_equalities))
            for (norm_idx, summation_idxs) in self.idx_level_equalities:
                eq_dict = {self.compmoment_from_idx[norm_idx]: 1}
                eq_dict.update(zip(
                    itemgetter(*summation_idxs)(self.compmoment_from_idx),
                    repeat(-1)
                ))
                self.minimal_equalities.append(eq_dict)

        self.minimal_inequalities = []
        self.moment_upperbounds  = {}
        self.moment_lowerbounds  = {m: 0. for m in self.hermitian_moments}

        self.set_objective(None)
        self.set_values(None)

        self.maskmatrices = {}
        self._relaxation_has_been_generated = True

    def set_extra_equalities(self,
                             extra_equalities: Union[list, None]) -> None:
        """Set extra equality constraints for the SDP.

        Parameters
        ----------
        extra_equalities : Union[list, None]
            List of additional equality constraints in the form of dictionaries
            (keys can be instances of `CompoundMonomial`, Symbols, strings, or
            integers), or SymPy expressions.
        """
        self.extra_equalities = []  # reset every time
        if not extra_equalities or extra_equalities is None:
            return
        self.extra_equalities = [self._sanitise_dict(eq)
                                 for eq in extra_equalities]

    def set_extra_inequalities(self,
                               extra_inequalities: Union[list, None]) -> None:
        """Set extra inequality constraints for the SDP.

        Parameters
        ----------
        extra_inequalities : Union[list, None]
            List of additional inequality constraints in the form of
            dictionaries (keys can be instances of `CompoundMonomial`, Symbols,
            strings, or integers) or SymPy expressions.
        """
        self.extra_inequalities = []  # reset every time
        if not extra_inequalities or extra_inequalities is None:
            return
        self.extra_inequalities = [self._sanitise_dict(ineq)
                                   for ineq in extra_inequalities]

    @property
    def moment_equalities(self) -> List[Dict]:
        """All equalities (minimal and extra) as one list of dictionaries."""
        return self.minimal_equalities + self.extra_equalities

    @property
    def moment_inequalities(self) -> List[Dict]:
        """All inequalities (minimal and extra) as one list of dictionaries."""
        return self.minimal_inequalities + self.extra_inequalities

    def set_bounds(self,
                   bounds: Union[dict, None],
                   bound_type: str = "up") -> None:
        r"""Set numerical lower or upper bounds on the moments generated in the
        SDP relaxation. The bounds are at the level of the SDP variables,
        and do not take into consideration non-convex constraints. E.g., two
        individual lower bounds, :math:`p_A(0|0) \geq 0.1` and
        :math:`p_B(0|0) \geq 0.1` do not directly impose the constraint
        :math:`p_A(0|0)*p_B(0|0) \geq 0.01`, which should be set manually if
        needed.

        Parameters
        ----------
        bounds : Union[dict, None]
            A dictionary with keys as monomials and values being the bounds.
            The keys can be either CompoundMonomial objects, or names (`str`)
            of Monomial objects.
        bound_type : str, optional
            Specifies whether we are setting upper (``"up"``) or lower
            (``"lo"``) bounds, by default "up".

        Examples
        --------
        >>> set_bounds({"pAB(00|00)": 0.2}, "lo")
        """
        assert bound_type in ["up", "lo"], \
            "The 'bound_type' argument should be either 'up' or 'lo'"
        if bounds is None:
            return
        # Sanitize list of bounds
        sanitized_bounds = {}
        for mon, bound in bounds.items():
            mon = self._sanitise_moment(mon)
            if mon not in sanitized_bounds.keys():
                sanitized_bounds[mon] = bound
            else:
                old_bound = sanitized_bounds[mon]
                assert np.isclose(old_bound, bound), \
                    (f"Contradiction: Cannot set the same monomial {mon} to " +
                     "have different upper bounds.")
        if bound_type == "up":
            self._reset_upperbounds()
            self.moment_upperbounds = sanitized_bounds
        else:
            self._reset_lowerbounds()
            self.moment_lowerbounds.update(sanitized_bounds)
        self._update_bounds(bound_type)

    @cached_property
    def atomic_factors(self):
        atoms = set()
        for mon in self.moments:
            atoms.update(mon.factors)
        return sorted(atoms)

    @cached_property
    def atomic_monomials(self):
        """Returns the atomic monomials."""
        return [self._monomial_from_atoms([atom]) for atom in self.atomic_factors]

    @cached_property
    def atom_from_name(self):
        """Returns the atomic monomials."""
        lookup_dict = {}
        for atom in self.atomic_factors:
            lookup_dict[atom.name] = atom
            lookup_dict[atom.legacy_name] = atom
        return lookup_dict

    @cached_property
    def atomic_monomials(self):
        """Returns the atomic monomials."""
        atoms = set()
        for mon in self.moments:
            atoms.update(mon.factors)
        return [self._monomial_from_atoms([atom]) for atom in sorted(atoms)]

    @cached_property
    def knowable_atoms(self):
        """Returns the knowable atoms."""
        return [m for m in self.atomic_monomials if m.is_knowable]

    @cached_property
    def do_conditional_atoms(self):
        """Returns the atomic monomials which correspond to do conditionals."""
        return [m for m in self.atomic_monomials if
                (m.is_do_conditional and not m.is_knowable)]

    @cached_property
    def physical_atoms(self):
        """Returns the knowable atoms."""
        return [m for m in self.atomic_monomials if m.is_all_commuting]

    def set_distribution(self,
                         prob_array: Union[np.ndarray, None],
                         use_lpi_constraints=False,
                         shared_randomness=False) -> None:
        r"""Set numerically all the knowable (and optionally semiknowable)
        moments according to the probability distribution specified.

        Parameters
        ----------
            prob_array : numpy.ndarray
                Multidimensional array encoding the distribution, which is
                called as ``prob_array[a,b,c,...,x,y,z,...]`` where
                :math:`a,b,c,\dots` are outputs and :math:`x,y,z,\dots` are
                inputs. Note: even if the inputs have cardinality 1 they must
                be specified, and the corresponding axis dimensions are 1.
                The parties' outcomes and measurements must be appear in the
                same order as specified by the ``order`` parameter in the
                ``InflationProblem`` used to instantiate ``InflationSDP``.

            use_lpi_constraints : bool, optional
                Specification whether linearized polynomial constraints (see,
                e.g., Eq. (D6) in `arXiv:2203.16543
                <http://www.arxiv.org/abs/2203.16543/>`_) will be imposed or
                not. By default ``False``.
            shared_randomness : bool, optional
                Specification whether higher order monomials may be calculated.
                If universal shared randomness is present (i.e., the flag is
                ``True``), only atomic monomials are assigned numerical values.
        """
        if prob_array is not None:
            assert prob_array.shape == self.expected_distro_shape, \
                f"Cardinalities mismatch: \n" \
                f"expected {self.expected_distro_shape}, \n " \
                f"got {prob_array.shape}"
            knowable_values = {atom: atom.compute_marginal(prob_array)
                               for atom in self.knowable_atoms}
        else:
            knowable_values = {}

        self.set_values(knowable_values,
                        use_lpi_constraints=use_lpi_constraints,
                        only_specified_values=shared_randomness)

    def set_objective(self,
                      objective: Union[sp.core.expr.Expr,
                                       dict,
                                       None],
                      direction: str = "max") -> None:
        """Set or change the objective function of the polynomial optimization
        problem.

        Parameters
        ----------
        objective : Union[sp.core.expr.Expr, dict, None]
            The objective function, either as a combination of sympy symbols,
            as a dictionary with keys the moments or their names, and as
            values the corresponding coefficients, or ``None`` for clearing
            a previous objective.
        direction : str, optional
            Direction of the optimization (``"max"``/``"min"``). By default
            ``"max"``.
        """
        assert (not self.supports_problem) or (objective is None), \
            "Supports problems do not support specifying objective functions."
        assert direction in ["max", "min"], ("The 'direction' argument should "
                                             + " be either 'max' or 'min'")

        self._reset_objective()
        if direction == "max":
            self.maximize = True
        else:
            self.maximize = False
        if objective is None:
            return
        else:
            if self.use_lpi_constraints and self.verbose > 0:
                warn("You have the flag `use_lpi_constraints` set to True. Be "
                     + "aware that imposing linearized polynomial constraints "
                     + "will constrain the optimization to distributions with "
                     + "fixed marginals.")
            sign = (1 if self.maximize else -1)
            self.objective = {mon: (sign * coeff) for mon, coeff
                              in self._sanitise_dict(objective).items()}
            self.objective.setdefault(self.One, 0)
            surprising_objective_terms = {mon for mon in self.objective.keys()
                                          if mon not in self.moments}
            assert len(surprising_objective_terms) == 0, \
                ("When interpreting the objective we have encountered at " +
                 "least one monomial that does not appear in the original " +
                 f"moment matrix:\n\t{surprising_objective_terms}")
            self._update_objective()

    def update_values(self,
                   values: Union[Dict[Union[CompoundMomentSDP,
                                            InternalAtomicMonomialSDP,
                                            sp.core.symbol.Symbol,
                                            str],
                                      Union[float, sp.core.expr.Expr]],
                                 None],
                   use_lpi_constraints: bool = False,
                   only_specified_values: bool = False) -> None:
        """Directly assign numerical values to variables in the moment matrix.
        This is done via a dictionary where keys are the variables to have
        numerical values assigned (either in their operator form, in string
        form, or directly referring to the variable in the moment matrix), and
        the values are the corresponding numerical quantities.

        Parameters
        ----------
        values : Union[None, Dict[Union[CompoundMomentSDP, InternalAtomicMonomialSDP,
                       sympy.core.symbol.Symbol, str], float]]
            The description of the variables to be assigned numerical values
            and the corresponding values. The keys can be either of the
            Monomial class, symbols or strings (which should be the name of
            some Monomial).

        use_lpi_constraints : bool
            Specification whether linearized polynomial constraints (see, e.g.,
            Eq. (D6) in arXiv:2203.16543) will be imposed or not.

        only_specified_values : bool
            Specifies whether one wishes to fix only the variables provided
            (``True``), or also the variables containing products of the
            monomials fixed (``False``). Regardless of this flag, unknowable
            variables can also be fixed.
        """
        if (values is None) or len(values) == 0:
            return

        self._reset_solution()

        self.use_lpi_constraints = use_lpi_constraints

        if (len(self.objective) > 1) and self.use_lpi_constraints:
            warn("You have an objective function set. Be aware that imposing "
                 + "linearized polynomial constraints will constrain the "
                 + "optimization to distributions with fixed marginals.")

        # It is funny to set values to moments created from operators that do
        # not commute with each other, so we display a warning.
        non_all_commuting_moments = set()
        for moment, value in values.items():
            try:  # We do this to deal with NaNs (such as in undefined supports)
                  # in a way that is also compatible with symbolic values.
                if np.isnan(value):
                    continue
            except TypeError:
                pass
            moment = self._sanitise_moment(moment)
            self.known_moments[moment] = value
            if (self.verbose > 0) and (not moment.is_all_commuting):
                non_all_commuting_moments.add(moment)
        if (len(non_all_commuting_moments) >= 1) and (self.verbose > 0):
            warn("When setting values, we encountered at least one monomial " +
                 "with noncommuting operators:\n\t" +
                 str(non_all_commuting_moments))
        del non_all_commuting_moments
        if not only_specified_values:
            atomic_knowns = {mon.factors[0]: val
                             for mon, val in self.known_moments.items()
                             if len(mon) == 1}
            atomic_knowns.update({atom.dagger: val
                                  for atom, val in atomic_knowns.items()})
            moments_not_present = set(self.known_moments.keys()
                                        ).difference(self.moments)
            for moment in moments_not_present:
                del self.known_moments[moment]

            # Get the remaining moments that need assignment
            if all(atom.is_knowable for atom in atomic_knowns):
                if not self.use_lpi_constraints:
                    remaining_mons = (mon for mon in self.moments
                                      if ((not mon.is_atomic)
                                          and mon.is_knowable))
                else:
                    remaining_mons = (mon for mon in self.moments
                                      if ((not mon.is_atomic)
                                          and mon.knowability_status
                                          in ["Knowable", "Semi"]))
            else:
                remaining_mons = (mon for mon in self.moments
                                  if not mon.is_atomic)
            surprising_semiknowns = set()
            for moment in remaining_mons:
                value, unknown_factors, known_status = moment.evaluate(
                    atomic_knowns,
                    self.use_lpi_constraints)
                if known_status == "Known":
                    self.known_moments[moment] = value
                elif known_status == "Semi":
                    if self.use_lpi_constraints:
                        unknown_mon = \
                            self._monomial_from_atoms(unknown_factors)
                        self.semiknown_moments[moment] = (value, unknown_mon)
                        if self.verbose > 0:
                            if unknown_mon not in self.moments:
                                surprising_semiknowns.add(unknown_mon)
                else:
                    pass
            if (len(surprising_semiknowns) >= 1) and (self.verbose > 0):
                warn("When processing LPI constraints we encountered at " +
                     "least one monomial that does not appear in the " +
                     f"original moment matrix:\n\t{surprising_semiknowns}")
            del atomic_knowns, surprising_semiknowns
        self._cleanup_after_set_values()

    def set_values(self, values, **kwargs):
        r"""Exactly like update_values, except it resets all known values to zero
        as an intermediate step
        """
        self._reset_values()
        if (values is None) or len(values) == 0:
            self._cleanup_after_set_values()
            return
        else:
            self.update_values(values, **kwargs)
            return

    def solve(self,
              interpreter="MOSEKFusion",
              feas_as_optim=False,
              solve_dual=True,
              solverparameters=None,
              solver_arguments={},
              verbose: int = -1) -> None:
        r"""Call a solver on the SDP relaxation. Upon successful solution, it
        returns the primal and dual objective values along with the solution
        matrices.

        Parameters
        ----------
        interpreter : str, optional
            The solver to be called. By default ``"MOSEKFusion"``.
        feas_as_optim : bool, optional
            Instead of solving the feasibility problem

                :math:`(1) \text{ find vars such that } \Gamma \succeq 0`

            setting this label to ``True`` solves instead the problem

                :math:`(2) \text{ max }\lambda\text{ such that }
                \Gamma - \lambda\cdot 1 \succeq 0.`

            The correspondence is that the result of (2) is positive if (1) is
            feasible, and negative otherwise. By default ``False``.
        solve_dual : bool, optional
            Optimize the dual problem (recommended). By default ``True``.
        solverparameters : dict, optional
            Extra parameters to be sent to the solver. By default ``None``.
        solver_arguments : dict, optional
            By default, solve will use the dictionary of SDP keyword arguments
            given by ``_prepare_solver_arguments()``. However, a user may
            manually override these arguments by passing their own here.
        verbose : int, optional
            How much information to display to the user. By default, ``-1``
            (which sets it to ``self.verbose``).
        """
        if not self._relaxation_has_been_generated:
            raise Exception("Relaxation is not generated yet. " +
                            "Call \"InflationSDP.get_relaxation()\" first")
        if feas_as_optim and len(self._processed_objective) > 1:
            warn("You have a non-trivial objective, but set to solve a " +
                 "feasibility problem as optimization. Setting "
                 + "feas_as_optim=False and optimizing the objective...")
            feas_as_optim = False

        real_verbose = self.verbose if verbose == -1 else verbose

        args = self._prepare_solver_arguments()
        args.update(solver_arguments)
        args.update({"feas_as_optim": feas_as_optim,
                     "verbose": real_verbose,
                     "solverparameters": solverparameters,
                     "solve_dual": solve_dual})

        self.solution_object = solveSDP_MosekFUSION(**args)

        self.status = self.solution_object["status"]
        if self.status == "optimal":
            self.success = True
            self.primal_objective = self.solution_object["primal_value"]
            self.objective_value  = self.solution_object["primal_value"]
            self.objective_value *= (1 if self.maximize else -1)
        else:
            self.success = False
            self.primal_objective = self.status
            self.objective_value  = self.status
        collect()

    ###########################################################################
    # PUBLIC ROUTINES RELATED TO THE PROCESSING OF CERTIFICATES               #
    ###########################################################################
    def certificate_as_dict(self,
                            clean: bool = True,
                            chop_tol: float = 1e-10,
                            round_decimals: int = 3) -> dict:
        """Give certificate as dictionary with monomials as keys and
        their coefficients in the certificate as the values. The certificate
        of incompatibility is ``cert < 0``.

        If the certificate is evaluated on a point giving a negative value, this
        guarantees that the compatibility test for the same point is infeasible
        provided the set of constraints of the program does not change. Warning:
        when using ``use_lpi_constraints=True`` the set of constraints depends
        on the specified distribution, thus the certificate is not guaranteed to
        apply.

        Parameters
        ----------
        clean : bool, optional
            If ``True``, eliminate all coefficients that are smaller than
            ``chop_tol``, normalise and round to the number of decimals
            specified by ``round_decimals``. By default ``True``.
        chop_tol : float, optional
            Coefficients in the dual certificate smaller in absolute value are
            set to zero. By default ``1e-10``.
        round_decimals : int, optional
            Coefficients that are not set to zero are rounded to the number of
            decimals specified. By default ``3``.

        Returns
        -------
        dict
            The expression of the certificate in terms of probabilities and
            marginals. The certificate of incompatibility is ``cert < 0``.
        """
        try:
            dual = self.solution_object["dual_certificate"]
        except AttributeError:
            raise Exception("For extracting a certificate you need to solve " +
                            "a problem. Call \"InflationSDP.solve()\" first.")
        if len(self.semiknown_moments) > 0:
            warn("Beware that, because the problem contains linearized " +
                 "polynomial constraints, the certificate is not guaranteed " +
                 "to apply to other distributions.")
        if np.allclose(list(dual.values()), 0.):
            return {}
        if clean:
            dual = clean_coefficients(dual, chop_tol, round_decimals)
        return {self.monomial_from_name[k]: v for k, v in dual.items()
                if not self.monomial_from_name[k].is_zero}

    def probs_from_dict(self,
                        dict_with_monomial_keys: dict) -> sp.core.add.Add:
        """Converts a monomial dictionary into a SymPy expression.

        Parameters
        ----------
        dict_with_monomial_keys : Dict[sympy.Symbol, float]
            Dictionary with monomials and associated coefficients.

        Returns
        -------
        sympy.core.add.Add
            The expression of the polynomial encoded in the dictionary.
        """
        polynomial = sp.S.Zero
        for mon, coeff in self._sanitise_dict(dict_with_monomial_keys).items():
            polynomial += coeff * mon.symbol
        return polynomial

    def certificate_as_probs(self,
                             clean: bool = True,
                             chop_tol: float = 1e-10,
                             round_decimals: int = 3) -> sp.core.add.Add:
        """Give certificate as symbolic sum of probabilities. The certificate
        of incompatibility is ``cert < 0``.

        If the certificate is evaluated on a point giving a negative value, this
        guarantees that the compatibility test for the same point is infeasible
        provided the set of constraints of the program does not change. Warning:
        when using ``use_lpi_constraints=True`` the set of constraints depends
        on the specified distribution, thus the certificate is not guaranteed to
        apply.

        Parameters
        ----------
        clean : bool, optional
            If ``True``, eliminate all coefficients that are smaller than
            ``chop_tol``, normalise and round to the number of decimals
            specified by ``round_decimals``. By default ``True``.
        chop_tol : float, optional
            Coefficients in the dual certificate smaller in absolute value are
            set to zero. By default ``1e-10``.
        round_decimals : int, optional
            Coefficients that are not set to zero are rounded to the number of
            decimals specified. By default ``3``.

        Returns
        -------
        sympy.core.add.Add
            The expression of the certificate in terms of probabilities and
            marginals. The certificate of incompatibility is ``cert < 0``.
        """
        return self.probs_from_dict(self.certificate_as_dict(
            clean=clean,
            chop_tol=chop_tol,
            round_decimals=round_decimals))

    def string_from_dict(self,
                         dict_with_monomial_keys: dict) -> str:
        """Converts a monomial dictionary into a string.

        Parameters
        ----------
        dict_with_monomial_keys : Dict[sympy.Symbol, float]
            Dictionary with monomials and associated coefficients.

        Returns
        -------
        str
            The expression of the certificate in string form.
        """
        as_dict = self._sanitise_dict(dict_with_monomial_keys)
        # Watch out for when "1" is note the same as "constant_term"
        constant_value = as_dict.pop(self.Constant_Term,
                                     as_dict.pop(self.One, 0.)
                                     )
        if constant_value:
            polynomial_as_str = str(constant_value)
        else:
            polynomial_as_str = ""
        for mon, coeff in as_dict.items():
            if mon.is_zero or np.isclose(np.abs(coeff), 0):
                continue
            else:
                polynomial_as_str += "+" if coeff >= 0 else "-"
                if np.isclose(abs(coeff), 1):
                    polynomial_as_str += mon.name
                else:
                    polynomial_as_str += "{0}*{1}".format(abs(coeff), mon.name)
        return polynomial_as_str[1:] if polynomial_as_str[
                                            0] == "+" else polynomial_as_str

    def certificate_as_string(self,
                              clean: bool = True,
                              chop_tol: float = 1e-10,
                              round_decimals: int = 3) -> str:
        """Give the certificate as a string of a sum of probabilities. The
        expression is in the form such that its satisfaction implies
        incompatibility.

        If the certificate is evaluated on a point giving a negative value, this
        guarantees that the compatibility test for the same point is infeasible
        provided the set of constraints of the program does not change. Warning:
        when using ``use_lpi_constraints=True`` the set of constraints depends
        on the specified distribution, thus the certificate is not guaranteed to
        apply.

        Parameters
        ----------
        clean : bool, optional
            If ``True``, eliminate all coefficients that are smaller than
            ``chop_tol``, normalise and round to the number of decimals
            specified by ``round_decimals``. By default, ``True``.
        chop_tol : float, optional
            Coefficients in the dual certificate smaller in absolute value are
            set to zero. By default, ``1e-10``.
        round_decimals : int, optional
            Coefficients that are not set to zero are rounded to the number of
            decimals specified. By default, ``3``.

        Returns
        -------
        str
            The certificate in terms of probabilities and marginals. The
            certificate of incompatibility is ``cert < 0``.
        """
        return self.string_from_dict(
            self.certificate_as_dict(
                clean=clean,
                chop_tol=chop_tol,
                round_decimals=round_decimals)) + " < 0"

    def evaluate_polynomial(self, polynomial: dict, prob_array: np.ndarray):
        """Evaluate the certificate of infeasibility in a target probability
        distribution. If the evaluation is a negative value, the distribution is
        not compatible with the causal structure. Warning: when using
        ``use_lpi_constraints=True`` the set of constraints depends on the
        specified distribution, thus the certificate is not guaranteed to apply.

        Parameters
        ----------
        polynomial : dict
            A dictionary of monomials with coefficients as values.
        prob_array : numpy.ndarray
            Multidimensional array encoding the distribution, which is
            called as ``prob_array[a,b,c,...,x,y,z,...]`` where
            :math:`a,b,c,\\dots` are outputs and :math:`x,y,z,\\dots` are
            inputs. Note: even if the inputs have cardinality 1 they must
            be specified, and the corresponding axis dimensions are 1.
            The parties' outcomes and measurements must appear in the
            same order as specified by the ``order`` parameter in the
            ``InflationProblem`` used to instantiate ``InflationLP``.

        Returns
        -------
        float
            The evaluation of the certificate of infeasibility in prob_array.
        """
        return sum((atom.compute_marginal(prob_array) * val
                    for atom, val in self._sanitise_dict(polynomial).items()))

    def evaluate_certificate(self, prob_array: np.ndarray) -> float:
        """Evaluate the certificate of infeasibility in a target probability
        distribution. If the evaluation is a negative value, the distribution is
        not compatible with the causal structure. Warning: when using
        ``use_lpi_constraints=True`` the set of constraints depends on the
        specified distribution, thus the certificate is not guaranteed to apply.

        Parameters
        ----------
        prob_array : numpy.ndarray
            Multidimensional array encoding the distribution, which is
            called as ``prob_array[a,b,c,...,x,y,z,...]`` where
            :math:`a,b,c,\\dots` are outputs and :math:`x,y,z,\\dots` are
            inputs. Note: even if the inputs have cardinality 1 they must
            be specified, and the corresponding axis dimensions are 1.
            The parties' outcomes and measurements must appear in the
            same order as specified by the ``order`` parameter in the
            ``InflationProblem`` used to instantiate ``InflationLP``.

        Returns
        -------
        float
            The evaluation of the certificate of infeasibility in prob_array.
        """
        if self.use_lpi_constraints:
            warn("You have used LPI constraints to obtain the certificate. " +
                 "Be aware that, because of that, the certificate may not be " +
                 "valid for other distributions.")
        return self.evaluate_polynomial(self.certificate_as_dict(), prob_array)

    def desymmetrize_certificate(self) -> dict:
        """If the scenario contains symmetries other than the inflation
        symmetries, this function writes a certificate of infeasibility valid
        for non-symmetric distributions too.

        Returns
        -------
        dict
            The expression of the un-symmetrized certificate in terms of
            probabilities and marginals. The certificate of incompatibility is
            ``cert < 0``.
        """
        try:
            dual = self.solution_object["dual_certificate"]
        except AttributeError:
            raise Exception("For extracting a certificate you need to solve " +
                            "a problem. Call \"InflationSDP.solve()\" first.")

        desymmetrized = {}
        norm = len(self.InflationProblem.symmetries)
        lexmon_names = self.InflationProblem._lexrepr_to_copy_index_free_names
        for symm in self.InflationProblem.symmetries:
            for mon, coeff in dual.items():
                mon = self.monomial_from_name[mon]
                if not mon.is_zero:
                    desymm_mon = lexmon_names[symm[mon.as_lexmon-1]]
                    desymm_mon = sorted(desymm_mon, key=lambda x: x[0])
                    if not mon.is_one:
                        desymm_name = "P[" + " ".join(desymm_mon) + "]"
                    else:
                        desymm_name = "1"
                    if desymm_name not in desymmetrized:
                        desymmetrized[desymm_name] = coeff / norm
                    else:
                        desymmetrized[desymm_name] += coeff / norm
        return desymmetrized

    ###########################################################################
    # OTHER ROUTINES EXPOSED TO THE USER                                      #
    ###########################################################################
    #TODO: Natively avoid intermediate 2d-representation in build_columns
    def build_columns(self,
                      column_specification: Union[str,
                                                  List[List[int]],
                                                  List[sp.core.symbol.Symbol]],
                      max_monomial_length: int = 0,
                      symbolic: bool = False) -> List[np.ndarray]:
        r"""Creates the objects indexing the columns of the moment matrix from
        a specification.

        Parameters
        ----------
        column_specification : Union[str, List[List[int]], List[sympy.core.symbol.Symbol]]
            See description in the ``self.generate_relaxation()`` method.
        max_monomial_length : int, optional
            Maximum number of letters in a monomial in the generating set,
            By default ``0``. Example: if we choose ``"local1"`` for
            three parties, it gives the set :math:`\{1, A, B, C, AB, AC, BC,
            ABC\}`. If we set ``max_monomial_length=2``, the generating set is
            instead :math:`\{1, A, B, C, AB, AC, BC\}`. By default ``0`` (no
            limit).
        symbolic: bool, optional
            If ``True``, it returns the columns as a list of sympy symbols
            parsable by `InflationSDP.generate_relaxation()`. By default
            ``False``.
        """
        columns = None
        if type(column_specification) == list:
            # There are three possibilities: list of lists, list of arrays 
            # or list of symbols. If list of lists, then it is the party
            # block encoding. If it is a list of arrays, then it can be either
            # monomials in the 2d encoding, or the 1d encoding.
            if type(column_specification[0]) == list:
                # This is the standard specification for the helper
                columns = self._build_cols_from_specs(column_specification)
            elif type(column_specification[0]) in {np.ndarray}:
                if len(np.array(column_specification[1]).shape) == 2:
                    # This is the 2d encoding, convert it to lexicographic repr
                    columns = [self.mon_to_lexrepr(mon)
                               for mon in column_specification]
                elif len(np.array(column_specification[1]).shape) == 1:
                    # This is the 1d encoding, make sure the dtype is correct
                    # for compatibility with numba
                    columns = [np.array(mon, dtype=np.intc)
                               for mon in column_specification]
                else:
                    raise Exception("The generating columns are not specified "
                                    + "in a valid format.")
            elif type(column_specification[0]) in [int, sp.core.symbol.Symbol,
                                                   sp.core.power.Pow,
                                                   sp.core.mul.Mul,
                                                   sp.core.numbers.One]:
                columns = []
                for col in column_specification:
                    if type(col) in [int, sp.core.numbers.One]:
                        if not np.isclose(float(col), 1):
                            raise Exception(f"Column {col} is just a number. "
                                            + "Please use a valid format.")
                        else:
                            columns.append(np.array([], dtype=np.intc))
                    elif type(col) in [sp.core.symbol.Symbol,
                                       sp.core.power.Pow,
                                       sp.core.mul.Mul]:
                        columns.append(self.mon_to_lexrepr(
                            self._interpret_name(col)))
                    else:
                        raise Exception(f"The column {col} is not specified " +
                                        "in a valid format.")
            else:
                raise Exception("The generating columns are not specified " +
                                "in a valid format.")
        elif type(column_specification) == str:
            if "npa" in column_specification.lower():
                npa_level = int(column_specification[3:])
                col_specs = [[]]
                if ((max_monomial_length > 0)
                        and (max_monomial_length < npa_level)):
                    max_length = max_monomial_length
                else:
                    max_length = npa_level
                for length in range(1, max_length + 1):
                    for number_tuple in product(
                            *[range(self.nr_parties)] * length
                                                ):
                        a = np.array(number_tuple)
                        # Add only if tuple is in increasing order
                        if np.all(a[:-1] <= a[1:]):
                            col_specs += [a.tolist()]
                columns = self._build_cols_from_specs(col_specs)

            elif (("local" in column_specification.lower())
                  or ("physical" in column_specification.lower())):
                lengths_init = (5 if "local" in column_specification.lower()
                                else 8)
                spec    = column_specification[:lengths_init]
                lengths = column_specification[lengths_init:]
                if len(lengths) == 0:
                    if spec == "local":
                        raise Exception("Please specify a precise local level")
                    else:
                        lengths = [None] * self.nr_parties
                elif len(lengths) == self.nr_parties:
                    lengths = [int(level) for level in lengths]
                else:
                    lengths = [int(lengths)] * self.nr_parties

                if spec == "local":
                    max_length = sum(lengths)
                    # Determine maximum length
                    if ((max_monomial_length > 0)
                            and (max_monomial_length < max_length)):
                        max_length = max_monomial_length

                    party_freqs = sorted((list(pfreq)
                                        for pfreq in product(
                                        *[range(level + 1) for level in lengths]
                                                            )
                                        if sum(pfreq) <= max_length),
                                        key=lambda x: (sum(x), [-p for p in x]))
                    col_specs = []
                    for pfreq in party_freqs:
                        operators = []
                        for party in range(self.nr_parties):
                            operators += [party] * pfreq[party]
                        col_specs += [operators]
                    columns = self._build_cols_from_specs(col_specs)
                else:
                    def template_to_event_boolarray(template: List[int],
                                                    decompressor: List[np.ndarray]) -> np.ndarray:
                        if len(template):
                            to_expand = partsextractor(decompressor, sorted(template, key=op_neg))
                            return reduce(nb_outer_bitwise_or, to_expand)
                        else:
                            return self.blank_bool_array
                    lex_idx_to_party = self.InflationProblem.party_from_templateidx
                    def template_is_short_enough(template: List[int]) -> bool:
                        parties_in_template = Counter(lex_idx_to_party[template])
                        for p, cap in enumerate(lengths):
                            if not cap == None:
                                if parties_in_template[p+1] > cap:
                                    return False
                        return True
                    relevant_physical_templates = list(filter(template_is_short_enough,
                                                              self.InflationProblem.all_and_maximal_compatible_templates(
                                                                  max_n=max_monomial_length,
                                                                  isolate_maximal=False)[0]))
                    physical_monomials_as_boolvecs = np.vstack([
                        template_to_event_boolarray(subclique, self._CG_limited_ortho_groups_as_boolarrays)
                        for subclique in relevant_physical_templates
                    ])
                    columns = sorted(
                        (np.flatnonzero(boolvec).astype(np.intc) + 1 
                        for boolvec in physical_monomials_as_boolvecs),
                                    key=lambda x: (len(x), tuple(x)))
            else:
                raise Exception("I have not understood the format of the "
                                + "column specification")
        else:
            raise Exception("I have not understood the format of the "
                            + "column specification")
        
        self.generating_monomials_1d = columns
        self.genmon_1d_to_index = {tuple(lexmon): i for i, lexmon in
                                   enumerate(self.generating_monomials_1d)}
        if len(self.genmon_1d_to_index) != len(self.generating_monomials_1d):
            self.generating_monomials_1d = \
                sorted(self.generating_monomials_1d, 
                       key=lambda x: (len(x), tuple(x)))
            self.genmon_1d_to_index = \
                {tuple(lexmon): i 
                 for i, lexmon in enumerate(self.generating_monomials_1d)}
            warn("The generating set of monomials included duplicate elements.")
        self.n_columns = len(self.generating_monomials_1d)
        output = self.generating_monomials_1d
        if symbolic:
            output = [reduce(sp.Mul,
                             self._lexrepr_to_symbols[lexmon],
                             sp.S.One)
                      for lexmon in self.generating_monomials_1d ]
        return output

    def reset(self, which: Union[str, List[str]] = "all") -> None:
        """Reset the various user-specifiable objects in the inflation SDP.

        Parameters
        ----------
        which : Union[str, List[str]]
            The objects to be reset. It can be fed as a single string or a list
            of them. Options include ``"bounds"``, ``"lowerbounds"``,
            ``"upperbounds"``, ``"objective"``, ``"values"``, and ``"all"``.
        """
        if type(which) == str:
            if which == "all":
                self.reset(["values", "bounds", "objective"])
            elif which == "bounds":
                self._reset_lowerbounds()
                self._reset_upperbounds()
            elif which == "lowerbounds":
                self._reset_lowerbounds()
            elif which == "upperbounds":
                self._reset_upperbounds()
            elif which == "objective":
                self._reset_objective()
            elif which == "values":
                self._reset_values()
            else:
                raise Exception(f"The attribute {which} is not part of " +
                                "InflationSDP.")
        else:
            for attr in which:
                self.reset(attr)
        collect()

    def write_to_file(self, filename: str) -> None:
        """Exports the problem to a file.

        Parameters
        ----------
        filename : str
            Name of the exported file. If no file format is specified, it
            defaults to sparse SDPA format. Supported formats are ``.mat``
            (MATLAB), ``.dat-s`` (SDPA), and ``.csv`` (human-readable).
        """
        # Determine file extension
        parts = filename.split(".")
        if len(parts) >= 2:
            extension = parts[-1]
        else:
            extension = "dat-s"
            filename += ".dat-s"

        # Write file according to the extension
        if self.verbose > 0:
            eprint("Writing the SDP program to", filename)
        if extension == "dat-s":
            write_to_sdpa(self, filename)
        elif extension == "csv":
            write_to_csv(self, filename)
        elif extension == "mat":
            write_to_mat(self, filename)
        else:
            raise Exception("File format not supported. Please choose between "
                            + "the extensions `.csv`, `.dat-s` and `.mat`.")

    ###########################################################################
    # ROUTINES RELATED TO CONSTRUCTING COMPOUND MONOMIAL INSTANCES            #
    ###########################################################################
    def _AtomicMonomial(self,
                        lexmon: np.ndarray) -> InternalAtomicMonomialSDP:
        """Construct an instance of the `InternalAtomicMonomialSDP` class from
        a 1D array description of a monomial.

        See the documentation of the `InternalAtomicMonomial` class for more
        details.

        Parameters
        ----------
        lexmon : numpy.ndarray
            Monomial encoded as a 1D array of integers, where integer
            represents the slots in the lexorder.

        Returns
        -------
        InternalAtomicMonomialSDP
            An instance of the `InternalAtomicMonomial` class representing the
            input 2D array monomial.
        """
        key = tuple(lexmon)
        try:
            return self.atomic_monomial_from_hash[key]
        except KeyError:
            repr_lexmon = self._to_inflation_repr_1d(lexmon)
            new_key      = tuple(repr_lexmon)
            try:
                mon = self.atomic_monomial_from_hash[new_key]
                self.atomic_monomial_from_hash[key] = mon
                return mon
            except KeyError:
                mon = InternalAtomicMonomialSDP(self, repr_lexmon)
                if self.real_qt:
                    conj = mon.dagger
                    mon = min(mon, conj)
                self.atomic_monomial_from_hash[key]     = mon
                self.atomic_monomial_from_hash[new_key] = mon
                return mon

    def Moment_2d(self, array2d: np.ndarray, idx=-1) -> CompoundMomentSDP:
        r"""Create an instance of the `CompoundMomentSDP` class from a 2D array.
        An instance of `CompoundMomentSDP` is a collection of
        `InternalAtomicMonomialSDP` instances.

        Parameters
        ----------
        array2d : numpy.ndarray
            Moment encoded as a 2D array of integers, where each row encodes
            one of the operators appearing in the moment.
        idx : int, optional
            Assigns an integer index to the resulting moment, which can be
            used as an id, by default -1.

        Returns
        -------
        CompoundMomentSDP
            The moment factorised into AtomicMonomials, all brought to
            representative form under inflation symmetries.

        Examples
        --------

        The moment
        :math:`\langle A^{0,2,1}_{x=2,a=3}C^{2,0,1}_{z=1,c=1}
        C^{1,0,2}_{z=0,c=0}\rangle` corresponds to the following 2D array:

        >>> m = np.array([[1, 0, 2, 1, 2, 3],
                          [3, 2, 0, 1, 1, 1],
                          [3, 1, 0, 2, 0, 0]])

        The resulting monomial, ``InflationSDP.Monomial(m)``, is a collection
        of two ``InternalAtomicMonomial`` s,
        :math:`\langle A^{0,1,1}_{x=2,a=3}C^{1,0,1}_{z=1,c=1}\rangle` and
        :math:`\langle C^{1,0,1}_{z=0,c=0}\rangle`, after factorizing the input
        monomial and reducing the inflation indices of each of the factors.
        """
        return self.Moment_1d(self.mon_to_lexrepr(array2d), idx=idx)

    def Moment_1d(self, lexmon: np.ndarray, idx=-1) -> CompoundMomentSDP:
        r"""Create an instance of the `CompoundMomentSDP` class from a 1D array.
        An instance of `CompoundMomentSDP` is a collection of
        `InternalAtomicMonomialSDP` instances.

        Parameters
        ----------
        lexmon : numpy.ndarray
            Moment encoded as a 1D array of integers, indicating positions in
            the _lexorder.
        idx : int, optional
            Assigns an integer index to the resulting moment, which can be
            used as an id, by default -1.

        Returns
        -------
        CompoundMomentSDP
            The moment factorised into AtomicMonomials, all brought to
            representative form under inflation symmetries.
        """
        # HACK: The lexorder of InflationProblem is different from that 
        # in InflationSDP!
        _factors = self.factorize_moment_1d(np.asarray(lexmon, dtype=np.intc)-1,
                                            canonical_order=False)
        list_of_atoms = [self._AtomicMonomial(factor + 1)
                         for factor in _factors if len(factor)]
        mon = self._monomial_from_atoms(list_of_atoms)
        mon.attach_idx(idx)
        return mon

    def _conjugate_lexmon(self,
                           lexmon: np.ndarray,
                           apply_only_commutations=True) -> np.ndarray:
        """Compute the canonical form of the conjugate of a monomial.

        Parameters
        ----------
        mon : numpy.ndarray
            Input monomial that cannot be further factorised in 1d format
        apply_only_commutations : bool, optional
            If ``True``, skip checking if monomial is zero and if there are
            square projectors.

        Returns
        -------
        numpy.ndarray
            The canonical form of the conjugate of the input monomial under
            relabelling through the inflation symmetries.
        """
        if self.all_commuting_q_1d(lexmon):
            return lexmon
        else:
            return self._to_inflation_repr_1d(reverse_mon(lexmon),
                                              apply_only_commutations)

    def _construct_mask_matrices(self) -> None:
        """Helper a function to associate each monomial appearing in the moment
        matrix with a unique mask matrix, as this is relevant to expressing an
        SDP in dual form.
        """
        if self._relaxation_has_been_generated:
            if self.n_columns > 0:
                self.maskmatrices = {
                    mon: sum(coo_array(self.momentmatrix == oldidx) for oldidx in oldidxs)
                    for mon, oldidxs in tqdm(self.old_indices_associated_with_monomial.items(),
                                    disable=not self.verbose,
                                    desc="Assigning mask matrices  ")
                                     }

    def _inflation_orbit_and_rep_1d(self, lexmon: np.ndarray):
        permuted_variants = np.take(self.lexorder_symmetries, lexmon, axis=1)
        permuted_variants = np.unique(permuted_variants, axis=0).astype(int)
        output_variants = \
            [tuple(self._to_canonical_memoized_1d(
                lexmon_variant, apply_only_commutations=True))
                           for lexmon_variant in permuted_variants]
        representative = np.array(min(output_variants), dtype=np.intc)
        return output_variants, representative

    def _monomial_from_atoms(self,
                             atoms: List[InternalAtomicMonomialSDP]
                             ) -> CompoundMomentSDP:
        """Build an instance of `CompoundMonomial` from a list of instances
        of `InternalAtomicMonomial`.

        Parameters
        ----------
        atoms : List[InternalAtomicMonomialSDP]
            List of instances of `InternalAtomicMonomial`.

        Returns
        -------
        CompoundMomentSDP
            A `CompoundMonomial` with atomic factors given by `atoms`.
        """
        list_of_atoms = []
        for factor in atoms:
            if factor.is_zero:
                list_of_atoms = [factor]
                break
            elif not factor.is_one:
                list_of_atoms.append(factor)
            else:
                pass
        atoms = tuple(sorted(list_of_atoms))
        if (not self.all_operators_commute) and (not self.real_qt):
            conjugate = [factor.dagger for factor in atoms]
            atoms = min(atoms, tuple(sorted(conjugate)))
            del conjugate
        try:
            mon = self.monomial_from_atoms[atoms]
            return mon
        except KeyError:
            mon = CompoundMomentSDP(atoms)
            try:
                mon.idx = self.first_free_idx
                self.first_free_idx += 1
            except AttributeError:
                pass
            self.monomial_from_atoms[atoms]   = mon
            self.monomial_from_name[mon.name] = mon
            self.monomial_from_name[mon.legacy_name] = mon  # For legacy compatibility!
            self.monomial_from_symbol[mon.symbol] = mon
            return mon

    def _sanitise_moment(self, moment: Any) -> CompoundMomentSDP:
        """Return a ``CompoundMonomial`` built from ``mon``, where ``mon`` can
        be either the name of a moment as a string, a SymPy variable, a
        monomial encoded as a 2D array, or an integer in case the moment is the
        unit moment or the zero moment.


        Parameters
        ----------
        moment : Any
            The name of a moment as a string, a SymPy variable with the name of
            a valid moment, a 2D array encoding of a moment or an integer in
            case the moment is the unit moment or the zero moment.

        Returns
        -------
        CompoundMomentSDP
            Instance of ``CompoundMonomial`` built from ``mon``.

        Raises
        ------
        Exception
            If ``mon`` is the constant monomial, it can only be numbers 0 or 1
        Exception
            If the type of ``mon`` is not supported.
        """
        if isinstance(moment, CompoundMomentSDP):
            return moment
        elif isinstance(moment, InternalAtomicMonomialSDP):
            return self._monomial_from_atoms([moment])
        elif isinstance(moment, (tuple, list, np.ndarray)):
            array = np.asarray(moment, dtype=self.np_dtype)
            assert array.ndim == 2, \
                "The monomial representations must be 2d arrays."
            assert array.shape[-1] == self._nr_properties, \
                "The input does not conform to the operator specification."
            canon_lexmon = self._to_canonical_memoized_1d(
                self.mon_to_lexrepr(array))
            return self.Moment_1d(canon_lexmon)
        elif isinstance(moment, (str, sp.core.symbol.Expr)):
            return self._sanitise_moment(self._interpret_name(moment))
        elif isinstance(moment, Real):
            if np.isclose(float(moment), 1):
                return self.One
            elif np.isclose(float(moment), 0):
                return self.Zero
            else:
                raise Exception(f"Constant monomial {moment} can only be 0 or 1.")
        else:
            raise Exception(f"sanitise_monomial: {moment} is of type " +
                            f"{type(moment)} and is not supported.")

    def _sanitise_dict(self, input_dict: Any) -> Dict:
        if isinstance(input_dict, sp.core.expr.Expr):
            if input_dict.free_symbols:
                input_dict_copy = {k: float(v) for k, v in sp.expand(
                    input_dict).as_coefficients_dict().items()}
            else:
                input_dict_copy = {}
        else:
            input_dict_copy = input_dict
        output_dict = defaultdict(int)
        for k, v in input_dict_copy.items():
            if not np.isclose(v, 0):
                output_dict[self._sanitise_moment(k)] += v
        return output_dict

    def _to_inflation_repr_1d(self,
                              lexmon: np.ndarray,
                              apply_only_commutations=False) -> np.ndarray:
        r"""Apply inflation symmetries to a monomial in order to bring it to
        its canonical form.

        Parameters
        ----------
        lexmon : numpy.ndarray
            Input monomial that cannot be further factorised.
        apply_only_commutations : bool, optional
            If ``True``, skip checking if monomial is zero and if there are
            multiple same projectors that square to just one of them.

        Returns
        -------
        numpy.ndarray
            The canonical form of the input monomial under relabelling through
            the inflation symmetries.
        """
        key = tuple(lexmon)
        if len(lexmon) == 0 or np.array_equiv(lexmon, 0):
            self.canonsym_lexmon_from_hash[key] = lexmon
            return lexmon
        else:
            pass
        try:
            return self.canonsym_lexmon_from_hash[key]
        except KeyError:
            pass
        canonical_mon = self._to_canonical_memoized_1d(lexmon,
                                                       apply_only_commutations)
        canonical_key = tuple(canonical_mon)
        try:
            repr_mon = self.canonsym_lexmon_from_hash[canonical_key]
            self.canonsym_lexmon_from_hash[key] = repr_mon
            return repr_mon
        except KeyError:
            pass

        other_keys, real_repr_lexmon = self._inflation_orbit_and_rep_1d(lexmon)
        other_keys.append(key)
        for key in other_keys:
            self.canonsym_lexmon_from_hash[key] = real_repr_lexmon
        return real_repr_lexmon

    ###########################################################################
    # ROUTINES RELATED TO NAME PARSING                                        #
    ###########################################################################
    
    def _interpret_name(self,
                        monomial: Union[str, sp.core.symbol.Expr, int]
                        ) -> np.ndarray:
        """Build a 2D array encoding of a monomial which can be passed either
        as a string, as a SymPy expression or as an integer.

        Parameters
        ----------
        monomial : Union[str, sympy.core.symbol.Expr, int]
            Input moment.

        Returns
        -------
        numpy.ndarray
            2D array encoding of the input moment.
        """
        if str(monomial) == '1':
            return self.identity_operator
        elif isinstance(monomial, str):
            try:
                return self.monomial_from_name[monomial]
            except KeyError:
                factors = monomial.split("*")
        elif isinstance(monomial, (tuple, list)):
            factors = [str(factor) for factor in monomial]
        elif isinstance(monomial, sp.core.symbol.Expr):
            try:
                return self.monomial_from_symbol[monomial]
            except KeyError:
                factors = [str(factor)
                           for factor in flatten_symbolic_powers(monomial)]
        else:
            raise Exception(f'Cannot interpret monomial with name {monomial} of type {type(monomial)}')
        return np.vstack(tuple(self._interpret_atomic_string(factor_string)
                               for factor_string in factors))

    def _interpret_atomic_string(self, factor_string: str) -> np.ndarray:
        """Build a 2D array encoding of a moment that cannot be further
        factorised into products of other moments.

        Parameters
        ----------
        factor_string : str
            String representation of a moment in expected value notation, e.g.,
            ``"<A_1_1_1_2*B_2_1_3_4>"``.

        Returns
        -------
        numpy.ndarray
            2D array encoding of the input atomic moment.
        """
        try:
            return self.atom_from_name[factor_string].as_2d_array
        except (KeyError, AttributeError):
            pass
        assert ((factor_string[0] == "<" and factor_string[-1] == ">")
                or (factor_string[0:1] == "P[" and factor_string[-1] == "]")
                or set(factor_string).isdisjoint(set("| "))), \
            ("Monomial names must be between < > signs, or in conditional " +
             f"probability form, whereas input received was {factor_string}")
        if factor_string[-1] in {'>', ')', "]", "}"}:
            cleaned_factor_string = factor_string[:-1]
            substrings_to_kill = {"P[", "P(", "p[", "p(", "<"}
            for substring in substrings_to_kill:
                cleaned_factor_string = cleaned_factor_string.replace(
                    substring, '')
            cleaned_factor_string = cleaned_factor_string.replace(' & ', ' ')
            operators = cleaned_factor_string.split(" ")
            return np.vstack(tuple(self._interpret_operator_string(op_string)
                                   for op_string in operators))
        else:
            return self._interpret_operator_string(factor_string)[np.newaxis]

    def _interpret_operator_string(self, op_string: str) -> np.ndarray:
        """Build a 1D array encoding of an operator passed as a string.

        Parameters
        ----------
        op_string : str
            String representation of an operator, e.g., ``"B_2_1_3_4"``.

        Returns
        -------
        numpy.ndarray
            2D array encoding of the operator.
        """
        return self._lexorder[self.op_from_name[op_string]]

    ###########################################################################
    # ROUTINES RELATED TO THE GENERATION OF THE MOMENT MATRIX                 #
    ###########################################################################
    
    def _build_cols_from_specs(self, col_specs: List[List[int]]) -> List:
        """Build the generating set for the moment matrix taking as input a
        block specified only the number of parties.

        For example, with ``col_specs=[[], [0], [2], [0, 2]]`` as input, we
        generate the generating set S={1, A_{inf}_xa, C_{inf'}_zc,
        A_{inf''}_x'a' * C{inf'''}_{z'c'}} where inf, inf', inf'' and inf'''
        represent all possible inflation copies indices compatible with the
        network structure, and x, a, z, c, x', a', z', c' are all possible
        input and output indices compatible with the cardinalities. As further
        examples, NPA level 2 for three parties is built from
        ``[[], [0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]``
        and "local level 1" for three parties is built from
        ``[[], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]``.

        Parameters
        ----------
        col_specs : List[List[int]]
            The column specification as specified in the method description.

        Returns
        -------
        List[numpy.ndarray]
            The list of operators indexing the columns, in array form.
        """
        if self.verbose > 1:
            # Display col_specs in a readable way
            to_print = []
            for specs in col_specs:
                to_print.append("1" if specs == []
                                else "".join([self.names[p] for p in specs]))
            eprint("Column structure:", "+".join(to_print))

        _zero_lexorder = np.array([0], dtype=np.intc)
        columns      = []
        seen_columns = set()
        for block in tqdm(col_specs, desc="Generating columns  ",
                          disable=not self.verbose):
            if block == []:
                _id = self.mon_to_lexrepr(self.identity_operator)
                seen_columns.add(tuple(_id))
                columns += [_id]
            else:
                meas_ops = [
                    np.nonzero(np.logical_and(
                        self._lexorder[:, 0] == party + 1,
                        self._lexorder[:, -1] != \
                            self.outcome_cardinalities[party] - 1))[0]
                                for party in block]
                for mon_lexrepr in product(*meas_ops):
                    canon = self._to_canonical_memoized_1d(mon_lexrepr)
                    if (not np.array_equal(canon, _zero_lexorder)
                        and len(canon) == len(block)):
                        _hash = tuple(canon)
                        if _hash not in seen_columns:
                            seen_columns.add(tuple(canon))
                            columns += [canon]

        return columns

    def _build_momentmatrix_1d_internal(self) -> Tuple[np.ndarray, Dict]:
        """Wrapper method for building the moment matrix."""
        problem_arr, canonical_lexmon_to_idx = \
            calculate_momentmatrix_1d_internal(self.generating_monomials_1d,
                                   self._notcomm,
                                   self._orthomat,
                                   commuting=self.all_operators_commute,
                                   verbose=self.verbose)
        idx_to_canonical_lexmon = {idx: np.asarray(lexmon, dtype=np.intc)
                                for (lexmon, idx) in
                                canonical_lexmon_to_idx.items()}
        return problem_arr, idx_to_canonical_lexmon

    def _discover_normalization_eqns(self) -> List[Tuple[int, List[int]]]:
        #TODO: Needs major overhaul to efficiently use 1d internal representation
        """Given the generating monomials, infer implicit normalization
        equalities between columns of the moment matrix. Each normalization
        equality is a two element tuple; the first element is an integer
        indicating a particular column of the moment matrix, the second element
        is a list of integers indicating other columns of the moment matrix
        such that the sum of the latter columns is equal to the former column.

        Returns
        -------
        List[Tuple[int, List[int]]]
            A list of normalization equalities between columns of the moment
        matrix.
        """

        # This will help us identify relevant operators with the last outcome
        first_outcome_boolmask = np.array(
            [self.has_children[op[0] - 1] and
              op[-1] == 0 for op in self._lexorder[1:]], dtype=bool)
        first_outcome_boolmask = np.hstack(([False], first_outcome_boolmask))
        
        # This will allow for easy substitution of operators with the last
        # outcome with the rest of the operators orthogonal to it
        lexmon_to_orthogroup = {}
        for group in self.InflationProblem._ortho_groups:
            first_outcome_op = \
                self.mon_to_lexrepr(np.expand_dims(group[0], axis=0))[0]
            lexmon_to_orthogroup[first_outcome_op] = \
                np.concatenate([self.mon_to_lexrepr(np.expand_dims(m, axis=0))
                                for m in group], dtype=np.intc)    
        
        column_level_equalities = []
        for i, lexmon in enumerate(self.generating_monomials_1d):
            last_outcome_ops = first_outcome_boolmask[lexmon]
            if last_outcome_ops.sum() > 0:
                eqs = []
                for i, op in enumerate(lexmon):
                    if last_outcome_ops[i]:
                        lhs = np.delete(lexmon, i)  # returns new copy
                        rhs = np.vstack((lexmon,)*len(lexmon_to_orthogroup[op]))
                        rhs[:, i] = lexmon_to_orthogroup[op]
                        eqs += [(lhs, list(rhs))]
                for eq in eqs:
                    try:
                        eq_idxs = [self.genmon_1d_to_index[tuple(eq[0])]]
                        eq_idxs.append([self.genmon_1d_to_index[tuple(m)]
                                        for m in eq[1]])
                        column_level_equalities += [tuple(eq_idxs)]
                    except KeyError:
                        break
        return column_level_equalities

    def _discover_columns_symmetries(self) -> np.ndarray:
        """Calculates all the symmetries and applies them to the set of
        operators used to define the moment matrix. The new set of operators
        is a permutation of the old. The function outputs a list of all
        permutations.

        Returns
        -------
        numpy.ndarray[int]
            The list of all permutations of the generating columns implied by
            the inflation symmetries.
        """
        discovered_symmetries = [np.arange(self.n_columns, dtype=int)]
        permutations_failed = 0
        permutation_failed = False
        for inf_sym in self.lexorder_symmetries[1:]:
            skip_this_one = False
            total_perm = np.empty(self.n_columns, dtype=int)
            for i, lexmon in enumerate(self.generating_monomials_1d):
                new_lexmon = np.argsort(inf_sym)[lexmon]
                new_lexmon_canon = self._to_canonical_memoized_1d(
                    new_lexmon,
                    apply_only_commutations=True)
                try:
                    total_perm[i] \
                    = self.genmon_1d_to_index[tuple(new_lexmon_canon)]
                except KeyError:
                    if self.verbose:
                        eprint(f"Warning: generating monomial before symmetry becomes unrecognizable after symmetry!")
                        eprint(f" Generating monomial before symmetry: {self._lexrepr_to_names[lexmon]}")
                        eprint(f" Generating monomial after symmetry: {self._lexrepr_to_names[new_lexmon_canon]}")
                    permutation_failed = True
                    permutations_failed += 1
                    skip_this_one = True
                    break
            if not skip_this_one:
                discovered_symmetries.append(total_perm)
        if permutation_failed and (self.verbose > 0):
            warn("The generating set is not closed under source swaps."
                 + f" {permutations_failed} symmetries were not be implemented.")
        return np.unique(discovered_symmetries, axis=0)[1:]

    ###########################################################################
    # HELPER FUNCTIONS FOR ENSURING CONSISTENCY                               #
    ###########################################################################

    def _cleanup_after_set_values(self) -> None:
        """Helper function to reset or make consistent class attributes after
        setting values."""
        if self.supports_problem:
            # Add lower bounds to monomials inside the support
            nonzero_known_monomials = [mon for
                                       mon, value in self.known_moments.items()
                                       if not np.isclose(value, 0)]
            for mon in nonzero_known_monomials:
                self.moment_lowerbounds[mon] = 1.
                del self.known_moments[mon]
            self.semiknown_moments = {}

        self._update_bounds("lo")
        self._update_bounds("up")
        self._update_objective()
        num_nontrivial_known = len(self.known_moments)
        if self.momentmatrix_has_a_zero:
            num_nontrivial_known -= 1
        if self.momentmatrix_has_a_one:
            num_nontrivial_known -= 1
        if self.verbose > 1 and num_nontrivial_known > 0:
            eprint("Number of variables with fixed numeric value:",
                  len(self.known_moments))
        if len(self.semiknown_moments):
            for k in self.known_moments.keys():
                self.semiknown_moments.pop(k, None)
        num_semiknown = len(self.semiknown_moments)
        if self.verbose > 1 and num_semiknown > 0:
            eprint(f"Number of semiknown variables: {num_semiknown}")

    def _reset_lowerbounds(self) -> None:
        """Reset the list of lower bounds."""
        self._reset_solution()
        self.moment_lowerbounds = {m: 0. for m in self.hermitian_moments}
        self._update_bounds("lo")

    def _reset_upperbounds(self) -> None:
        """Reset the list of upper bounds."""
        self._reset_solution()
        self.moment_upperbounds = {}

    def _reset_objective(self) -> None:
        """Reset the objective function."""
        self._reset_solution()
        self.objective = {self.One: 0.}
        self._processed_objective = self.objective
        self.maximize = True  # Direction of the optimization

    def _reset_values(self) -> None:
        """Reset the known values."""
        self._reset_solution()
        self.known_moments     = {}
        self.semiknown_moments = {}
        if self.momentmatrix_has_a_zero:
            self.known_moments[self.Zero] = 0.
        self.known_moments[self.One] = 1.
        self.extra_equalities = []
        self.extra_inequalities = []
        collect()

    def _update_objective(self) -> None:
        """Process the objective with the information from known_moments
        and semiknown_moments.
        """
        self._processed_objective = self.objective.copy()
        knowns_to_process = set(self.known_moments.keys()
                                ).intersection(
                                    self._processed_objective.keys())
        knowns_to_process.discard(self.One)
        for m in knowns_to_process:
            value = self.known_moments[m]
            self._processed_objective[self.One] += \
                self._processed_objective[m] * value
            del self._processed_objective[m]
        semiknowns_to_process = set(self.semiknown_moments.keys()
                                    ).intersection(
                                        self._processed_objective.keys())
        for mon in semiknowns_to_process:
            coeff = self._processed_objective[mon]
            for (subs_coeff, subs) in self.semiknown_moments[mon]:
                self._processed_objective[subs] = \
                    self._processed_objective.get(subs, 0) + coeff * subs_coeff
                del self._processed_objective[mon]
        collect()

    def _update_bounds(self, typ: str) -> None:
        """Helper function to check that bounds are consistent with the
        specified known values.

        Parameters
        ----------
        typ : str
            Specification of upper (`"up"`) or lower (`"lo"`) bounds.
        """
        if typ == "up":
            bounds = self.moment_upperbounds
            dir = "upp"
        elif typ == "lo":
            bounds = self.moment_lowerbounds
            dir = "low"
        else:
            raise Exception(f"The bound type was {typ}, but it must be " +
                            "either \"up\" or \"lo\".")
        for mon, value in self.known_moments.items():
            if isinstance(value, Real):
                try:
                    b = bounds[mon]
                    condition = (b >= value) if typ == "up" else (b <= value)
                    assert condition, (f"Value {value} assigned for " +
                                       f"monomial {mon} contradicts the " +
                                       f"assigned {dir}er bound of {b}.")
                    del bounds[mon]
                except KeyError:
                    pass

    ###########################################################################
    # OTHER ROUTINES                                                          #
    ###########################################################################
    
    def _atomic_knowable_q(self, atomic_monarray: np.ndarray) -> bool:
        """Return ``True`` if the input monomial, encoded as a 2D array,
        can be associated to a knowable value in the scenario, and ``False``
        otherwise.

        Parameters
        ----------
        atomic_monarray : numpy.ndarray
            Monomial encoded as a 2D array.

        Returns
        -------
        bool
            Whether the monomial could be assigned a numerical value.
        """
        if not is_knowable(atomic_monarray):
            return False
        elif self.network_scenario:
            return True
        else:
            return self._is_knowable_q_non_networks(np.take(atomic_monarray,
                                                            [0, -2, -1],
                                                    axis=1))
    def _prepare_solver_arguments(self) -> dict:
        """Prepare arguments to pass to the solver.

        The solver takes as input the following arguments, which are all
        dicts with keys as scalar SDP variables:
            * "mask_matrices": dict with values the binary matrices with the
            positions of the keys in the moment matrix.
            * "objective": dict with values the coefficient of the key
            variable in the objective function.
            * "known_vars": scalar variables that are fixed to be constant.
            * "semiknown_vars": if applicable, linear proportionality
            constraints between variables in the SDP.
            * "equalities": list of dicts where each dict gives the
            coefficients of the keys in a linear equality constraint.
            * "inequalities": list of dicts where each dict gives the
            coefficients of the keys in a linear inequality constraint.

        Returns
        -------
        dict
            A tuple with the arguments to be passed to the solver.

        Raises
        ------
        Exception
            If the SDP relaxation has not been calculated yet.
        """
        if not self._relaxation_has_been_generated:
            raise Exception("Relaxation is not generated yet. " +
                            "Call \"InflationSDP.get_relaxation()\" first")

        assert set(self.known_moments.keys()).issubset(self.moments),\
            ("Error: Tried to assign known values outside of moment matrix: " +
             str(set(self.known_moments.keys()
                     ).difference(self.moments)))
        if len(self.maskmatrices) == 0:
            self._construct_mask_matrices()
        solverargs = {"mask_matrices": {mon.name: mask_matrix
                                        for mon, mask_matrix
                                        in self.maskmatrices.items()},
                      "objective": {mon.name: coeff for mon, coeff
                                    in self._processed_objective.items()},
                      "known_vars": {mon.name: val for mon, val
                                     in self.known_moments.items()},
                      "semiknown_vars": {mon.name: (coeff, subs.name)
                                         for mon, (coeff, subs)
                                         in self.semiknown_moments.items()},
                      "equalities": [{mon.name: coeff
                                          for mon, coeff in eq.items()}
                                     for eq in self.minimal_equalities],
                      "inequalities": [{mon.name: coeff
                                        for mon, coeff in ineq.items()}
                                       for ineq in self.minimal_inequalities]
                      }
        # Add the constant 1 in case of unnormalized problems removed it
        solverargs["known_vars"][self.constant_term_name] = 1.
        for mon, bnd in self.moment_lowerbounds.items():
            lb = {mon.name: 1}
            if not np.isclose(bnd, 0):
                lb[self.constant_term_name] = -bnd
            solverargs["inequalities"].append(lb)
        for mon, bnd in self.moment_upperbounds.items():
            ub = {mon.name: -1}
            if not np.isclose(bnd, 0):
                ub[self.constant_term_name] = bnd
            solverargs["inequalities"].append(ub)
        solverargs["mask_matrices"][self.constant_term_name] = coo_array(
            (self.n_columns, self.n_columns))
        return solverargs

    def _reset_solution(self) -> None:
        """Resets class attributes storing the solution to the SDP
        relaxation."""
        for attribute in {"primal_objective",
                          "objective_value",
                          "solution_object"}:
            try:
                delattr(self, attribute)
            except AttributeError:
                pass
        self.status = "Not yet solved"

    def _to_canonical_memoized_1d(self,
                                  lexmon: np.ndarray,
                                  apply_only_commutations=False) -> np.ndarray:
        """Cached function to convert a monomial to its canonical form.

        It checks whether the input monomial's canonical form has already been
        calculated and stored in the ``InflationSDP.canon_ndarray_from_hash``.
        If not, it calculates it.

        Parameters
        ----------
        lexmon : numpy.ndarray
            Moment encoded as a 1D array.
        apply_only_commutations : bool, optional
            If ``True``, skip the removal of projector squares and the test to
            see if the monomial is equal to zero, by default ``False``.

        Returns
        -------
        numpy.ndarray
            Moment in canonical form as 1D array.
        """
        key = tuple(lexmon)
        try:
            return self.canon_lexmon_from_hash[key]
        except KeyError:
            if len(lexmon) == 0 or np.array_equiv(lexmon, 0):
                self.canon_lexmon_from_hash[key] = lexmon
                return lexmon
            else:
                new_lexmon = \
                    to_canonical_1d_internal(
                        np.asarray(lexmon, dtype=np.int32),
                        self._notcomm, self._orthomat, 
                        self.all_operators_commute,
                        apply_only_commutations=apply_only_commutations)
                new_key = tuple(new_lexmon)
                self.canon_lexmon_from_hash[key]     = new_lexmon
                self.canon_lexmon_from_hash[new_key] = new_lexmon
                return new_lexmon

    def mon_to_lexrepr(self, mon: np.ndarray) -> np.ndarray:
        """Convert a monomial from 2D array form to its lexicographic form.

        In the 2D array form, rows represent operators and columns represent
        properties. In the lexicographic form, each entry represents the index
        of the operator in the lexicographic order.
        
        Example: ``[[1, 0, 1], [2, 0, 0], [1, 1, 0]]``, with 
        ``lexorder=[[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [2, 0, 0]]``
        is converted to ``[1, 4, 2]``.

        Parameters
        ----------
        mon : np.ndarray
            Monomial in 2D array form.

        Returns
        -------
        np.ndarray
            Monomial in the 1D lexicographic form.
        """
        template = np.empty(len(mon), dtype=object)
        template[:] = np.asarray(mon, self.np_dtype).ravel().view(self._astuples_dtype)
        return np.array(partsextractor(self.op_to_lexrepr_dict, template), dtype=np.intc)
