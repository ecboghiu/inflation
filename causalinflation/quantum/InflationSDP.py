"""
The module generates the semidefinite program associated to a quantum inflation
instance (see arXiv:1909.10519).

@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""
import numpy as np
import sympy as sp

from collections import Counter, deque
from functools import reduce
from gc import collect
from itertools import chain, count, product, permutations
from numbers import Real
from scipy.sparse import lil_matrix
from typing import List, Dict, Tuple, Union, Any
from warnings import warn

from causalinflation import InflationProblem
from .fast_npa import (all_commuting_test,
                       apply_source_perm,
                       commutation_matrix,
                       nb_mon_to_lexrepr,
                       reverse_mon,
                       to_canonical)
from .monomial_classes import InternalAtomicMonomial, CompoundMonomial
from .quantum_tools import (apply_inflation_symmetries,
                            calculate_momentmatrix,
                            clean_coefficients,
                            construct_normalization_eqs,
                            factorize_monomial,
                            flatten_symbolic_powers,
                            format_permutations,
                            generate_operators,
                            is_knowable,
                            party_physical_monomials,
                            reduce_inflation_indices)
from .sdp_utils import solveSDP_MosekFUSION
from .writer_utils import write_to_csv, write_to_mat, write_to_sdpa
from ..utils import flatten

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
    supports_problem : bool, optional
        Whether to consider feasibility problems with distributions, or just
        with the distribution's support. By default ``False``.
    verbose : int, optional
        Optional parameter for level of verbose:

            * 0: quiet (default),
            * 1: monitor level: track program process,
            * 2: debug level: show properties of objects created.
    """
    constant_term_name = "constant_term"

    def __init__(self, inflationproblem: InflationProblem,
                 commuting: bool = False,
                 supports_problem: bool = False,
                 verbose: int = 0):
        """Constructor for the InflationSDP class.
        """
        self.supports_problem = supports_problem
        if inflationproblem.verbose > verbose:
            print("Overriding the verbosity parameter from InflationProblem")
        self.verbose = verbose
        self.commuting = commuting
        self.InflationProblem = inflationproblem
        self.names = self.InflationProblem.names
        self.names_to_ints = {name: i + 1 for i, name in enumerate(self.names)}
        if self.verbose > 1:
            print(self.InflationProblem)

        self.nr_parties = len(self.names)
        self.nr_sources = self.InflationProblem.nr_sources
        self.hypergraph = self.InflationProblem.hypergraph
        self.inflation_levels = \
            self.InflationProblem.inflation_level_per_source
        self.has_children = self.InflationProblem.has_children
        self.outcome_cardinalities = \
            self.InflationProblem.outcomes_per_party.copy()
        if self.supports_problem:
            self.has_children = np.ones(self.nr_parties, dtype=int)
        else:
            self.has_children = self.InflationProblem.has_children
        self.outcome_cardinalities += self.has_children
        self.setting_cardinalities = self.InflationProblem.settings_per_party

        might_have_a_zero = np.any(self.outcome_cardinalities > 1)

        self.measurements = self._generate_parties()
        if self.verbose > 1:
            print("Number of single operator measurements per party:", end="")
            prefix = " "
            for i, measures in enumerate(self.measurements):
                counter = count()
                deque(zip(chain.from_iterable(
                    chain.from_iterable(measures)),
                          counter),
                      maxlen=0)
                print(prefix + f"{self.names[i]}={next(counter)}", end="")
                prefix = ", "
            print()
        self.use_lpi_constraints = False
        self.network_scenario    = self.InflationProblem.is_network
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
        self.identity_operator = np.empty((0, self._nr_properties),
                                          dtype=self.np_dtype)
        self.zero_operator = np.zeros((1, self._nr_properties),
                                      dtype=self.np_dtype)

        # Define default lexicographic order through np.lexsort
        lexorder = self._interpret_name(flatten(self.measurements))
        if might_have_a_zero:
            lexorder = np.concatenate((self.zero_operator, lexorder))

        self._default_lexorder = lexorder[np.lexsort(np.rot90(lexorder))]
        self._lexorder = self._default_lexorder.copy()

        self._default_notcomm = commutation_matrix(self._lexorder,
                                                   self.commuting)
        self._notcomm = self._default_notcomm.copy()

        self.canon_ndarray_from_hash    = dict()
        self.canonsym_ndarray_from_hash = dict()
        self.atomic_monomial_from_hash  = dict()
        self.monomial_from_atoms        = dict()
        self.monomial_from_name         = dict()
        self.Zero = self.Monomial(self.zero_operator, idx=0)
        self.One  = self.Monomial(self.identity_operator, idx=1)
        self._relaxation_has_been_generated = False

    ###########################################################################
    # MAIN ROUTINES EXPOSED TO THE USER                                       #
    ###########################################################################
    def generate_relaxation(self,
                            column_specification:
                            Union[str,
                                  List[List[int]],
                                  List[sp.core.symbol.Symbol]] = "npa1"
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
              the set :math:`{1, A, B, AA, AB, BB, AAA, AAB, ABB, BBB\}` for
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
              the measurement operators in `self.measurements`. This list needs
              to have the identity ``sympy.S.One`` as the first element.
        """
        generating_monomials = self.build_columns(column_specification)
        # Generate dictionary to indices (used in dealing with symmetries and
        # column-level equalities)
        genmon_hash_to_index = {self._from_2dndarray(op): i
                                for i, op in enumerate(generating_monomials)}
        # Check for duplicates
        if len(genmon_hash_to_index) < len(generating_monomials):
            generating_monomials = [generating_monomials[i]
                                    for i in genmon_hash_to_index.values()]
            genmon_hash_to_index = {hash: i for i, hash
                                    in enumerate(genmon_hash_to_index.keys())}
            if self.verbose > 0:
                warn("Duplicates were detected in the list of generating " +
                     "monomials and automatically removed.")
        self.genmon_hash_to_index = genmon_hash_to_index
        self.n_columns            = len(generating_monomials)
        self.generating_monomials = generating_monomials
        del generating_monomials, genmon_hash_to_index
        collect()
        if self.verbose > 0:
            print("Number of columns in the moment matrix:", self.n_columns)

        # Calculate the moment matrix without the inflation symmetries
        unsymmetrized_mm, unsymmetrized_corresp = self._build_momentmatrix()
        symmetrization_required = np.any(self.inflation_levels - 1)
        additional_var = 0
        if self.verbose > 1:
            extra_msg = (" before symmetrization" if symmetrization_required
                         else "")
            if 0 in unsymmetrized_mm.flat:
                additional_var = 1
            print("Number of variables" + extra_msg + ":",
                  len(unsymmetrized_corresp) + additional_var)

        # Calculate the inflation symmetries
        self.inflation_symmetries = self._discover_inflation_symmetries()

        # Apply the inflation symmetries to the moment matrix
        self.momentmatrix, self.orbits, representative_unsym_idxs = \
            apply_inflation_symmetries(unsymmetrized_mm,
                                       self.inflation_symmetries,
                                       self.verbose)
        self.symmetrized_corresp = \
            {self.orbits[idx]: unsymmetrized_corresp[idx]
             for idx in representative_unsym_idxs.flat if idx >= 1}
        unsymidx_from_hash = {self._from_2dndarray(mon): idx for (idx, mon) in
                              unsymmetrized_corresp.items()
                              if all_commuting_test(mon,
                                                    self._lexorder,
                                                    self._notcomm)}
        for (hash, idx) in unsymidx_from_hash.items():
            self.canonsym_ndarray_from_hash[hash] = \
                self.symmetrized_corresp[self.orbits[idx]]
        if self.verbose > 0:
            extra_msg = (" after symmetrization" if symmetrization_required
                         else "")
            print(f"Number of variables{extra_msg}: "
                  + f"{len(self.symmetrized_corresp)+additional_var}")
        del unsymidx_from_hash, unsymmetrized_mm, unsymmetrized_corresp, \
            symmetrization_required, additional_var
        # This is a good time to reclaim memory, as unsymmetrized_mm can be GBs
        collect()

        self.momentmatrix_has_a_zero, self.momentmatrix_has_a_one = \
            np.in1d([0, 1], self.momentmatrix.ravel())



        # Associate Monomials to the remaining entries. The zero monomial is
        # not stored during calculate_momentmatrix
        self.compmonomial_from_idx = dict()
        if self.momentmatrix_has_a_zero:
            self.compmonomial_from_idx[0] = self.Zero
        for (idx, mon) in tqdm(self.symmetrized_corresp.items(),
                           disable=not self.verbose,
                           desc="Initializing monomials   "):
            self.compmonomial_from_idx[idx] = self.Monomial(mon, idx)

        self.monomials = list(self.compmonomial_from_idx.values())
        assert all(v == 1 for v in Counter(self.monomials).values()), \
            "Multiple indices are being associated to the same monomial"
        knowable_atoms = set()
        for mon in self.monomials:
            knowable_atoms.update(mon.knowable_factors)
        self.knowable_atoms = [self._monomial_from_atoms([atom])
                               for atom in knowable_atoms]
        del knowable_atoms

        _counter = Counter([mon.knowability_status for mon in self.monomials])
        self.n_knowable           = _counter["Knowable"]
        self.n_something_knowable = _counter["Semi"]
        self.n_unknowable         = _counter["Unknowable"]
        if self.verbose > 1:
            print(f"The problem has {self.n_knowable} knowable monomials, " +
                  f"{self.n_something_knowable} semi-knowable monomials, " +
                  f"and {self.n_unknowable} monomials.")

        if self.commuting:
            self.physical_monomials = self.monomials
        else:
            self.physical_monomials = list()
            for mon in self.monomials:
                if mon.is_physical:
                    self.physical_monomials.append(mon)
            if self.verbose > 1:
                print(f"The problem has {len(self.physical_monomials)} " +
                      "non-negative monomials.")

        # This dictionary useful for certificates_as_probs
        self.names_to_symbols = {mon.name: mon.symbol
                                 for mon in self.monomials}
        self.names_to_symbols[self.constant_term_name] = sp.S.One

        # In non-network scenarios we do not use Collins-Gisin notation for
        # some variables, so there exist normalization constraints between them
        if self.network_scenario:
            self.moment_equalities = []
        else:
            self.column_level_equalities = self._discover_column_equalities()
            self.idx_level_equalities    = construct_normalization_eqs(
                                                self.column_level_equalities,
                                                self.momentmatrix,
                                                self.verbose)
            if self.verbose > 1 and len(self.idx_level_equalities):
                print("Number of normalization equalities:",
                      len(self.idx_level_equalities))
            self.moment_equalities = [{self.compmonomial_from_idx[idx]: coeff
                                       for idx, coeff in eq.items()}
                                      for eq in self.idx_level_equalities]

        self.moment_inequalities = []
        self.moment_upperbounds  = dict()
        self.moment_lowerbounds  = {m: 0. for m in self.physical_monomials}

        self._set_lowerbounds(None)
        self._set_upperbounds(None)
        self.set_objective(None)
        self.set_values(None)

        # Get mask matrices associated with each monomial
        for mon in tqdm(self.monomials,
                           disable=not self.verbose,
                           desc="Assigning mask matrices  "):
            mon.mask_matrix = lil_matrix(self.momentmatrix == mon.idx)
        self.maskmatrices   = {mon: mon.mask_matrix for mon in self.monomials}

        self._relaxation_has_been_generated = True

    def set_bounds(self,
                   bounds: Union[dict, None],
                   bound_type: str = "up") -> None:
        """Set numerical lower or upper bounds on the moments generated in
        the SDP relaxation. 
        
        These bounds are at the level of the SDP variables without taking into
        consideration non-convex constraints. E.g., two individual lower bounds
        `{'pA(0|0)': 0.1, 'pB(0|0)': 0.1}` are not equivalent to
        `{'pA(0|0)*pB(0|0)': 0.01}`. The latter lower bound needs to be set
        manually. More complex bounds, e.g., in terms of other variables, should
        be set as inequality constraints. 

        Parameters
        ----------
        bounds : Union[dict, None]
            A dictionary with keys as monomials and values being the bounds.
            The keys can be either CompoundMonomial objects, or names (`str`)
            of Monomial objects.
        bound_type : str, optional
            Specifies whether we are setting upper ("up") or lower ("lo")
            bounds, by default "up".
            
        Examples
        --------
        >>> set_bounds({'pAB(00|00)': 0.2}, 'lo')
        True
        """
        assert bound_type in ["up", "lo"], \
            "The 'bound_type' argument should be either 'up' or 'lo'"
        if bound_type == "up":
            self._set_upperbounds(bounds)
        else:
            self._set_lowerbounds(bounds)

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
            knowable_values = {atom: atom.compute_marginal(prob_array)
                               for atom in self.knowable_atoms}
        else:
            knowable_values = dict()

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
            as a dictionary with keys the monomials or their names, and as
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
        elif isinstance(objective, sp.core.expr.Expr):
            if objective.free_symbols:
                objective_raw = sp.expand(objective).as_coefficients_dict()
                objective_raw = {k: float(v)
                                 for k, v in objective_raw.items()}
            else:
                objective_raw = {self.One: float(objective)}
            return self.set_objective(objective_raw, direction)
        else:
            if self.use_lpi_constraints and self.verbose > 0:
                warn("You have the flag `use_lpi_constraints` set to True. Be "
                     + "aware that imposing linearized polynomial constraints "
                     + "will constrain the optimization to distributions with "
                     + "fixed marginals.")
            sign = (1 if self.maximize else -1)
            objective_dict = {self.One: 0}
            for mon, coeff in objective.items():
                if not np.isclose(coeff, 0):
                    mon = self._sanitise_monomial(mon)
                    objective_dict[mon] = \
                        objective_dict.get(mon, 0) + (sign * coeff)
            self.objective = objective_dict
            surprising_objective_terms = {mon for mon in self.objective.keys()
                                          if mon not in self.monomials}
            assert len(surprising_objective_terms) == 0, \
                ("When interpreting the objective we have encountered at " +
                 "least one monomial that does not appear in the original " +
                 f"moment matrix:\n\t{surprising_objective_terms}")
            self._update_objective()

    def set_values(self,
                   values: Union[None, Dict[Union[CompoundMonomial,
                                                  InternalAtomicMonomial,
                                                  sp.core.symbol.Symbol,
                                                  str],
                                            float]],
                   use_lpi_constraints=False,
                   only_specified_values=False) -> None:
        """Directly assign numerical values to variables in the moment matrix.
        This is done via a dictionary where keys are the variables to have
        numerical values assigned (either in their operator form, in string
        form, or directly referring to the variable in the moment matrix), and
        the values are the corresponding numerical quantities.

        Parameters
        ----------
        values : Union[None, Dict[Union[CompoundMonomial, InternalAtomicMonomial, sympy.core.symbol.Symbol, str], float]]
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
        self._reset_values()

        if (values is None) or (len(values) == 0):
            self._cleanup_after_set_values()
            return

        self.use_lpi_constraints = use_lpi_constraints

        if (len(self.objective) > 1) and self.use_lpi_constraints:
            warn("You have an objective function set. Be aware that imposing "
                 + "linearized polynomial constraints will constrain the "
                 + "optimization to distributions with fixed marginals.")

        # It is funny to set values to monomials created from operators that do
        # not commute with each other, so we display a warning.
        non_all_commuting_monomials = set()
        for mon, value in values.items():
            if not np.isnan(value):
                mon = self._sanitise_monomial(mon)
                self.known_moments[mon] = value
                if (self.verbose > 0) and (not mon.is_all_commuting):
                    non_all_commuting_monomials.add(mon)
        if (len(non_all_commuting_monomials) >= 1) and (self.verbose > 0):
            warn("When setting values, we encountered at least one monomial " +
                 "with noncommuting operators:\n\t" +
                 str(non_all_commuting_monomials))
        del non_all_commuting_monomials
        if not only_specified_values:
            atomic_knowns = {mon.knowable_factors[0]: val
                             for mon, val in self.known_moments.items()
                             if len(mon) == 1}
            atomic_knowns.update({atom.dagger: val
                                  for atom, val in atomic_knowns.items()})
            monomials_not_present = set(self.known_moments.keys()
                                        ).difference(self.monomials)
            for mon in monomials_not_present:
                del self.known_moments[mon]

            # Get the remaining monomials that need assignment
            if all(atom.is_knowable for atom in atomic_knowns):
                if not self.use_lpi_constraints:
                    remaining_mons = (mon for mon in self.monomials
                                      if ((not mon.is_atomic)
                                          and mon.is_knowable))
                else:
                    remaining_mons = (mon for mon in self.monomials
                                      if ((not mon.is_atomic)
                                          and mon.knowability_status
                                          in ["Knowable", "Semi"]))
            else:
                remaining_mons = (mon for mon in self.monomials
                                  if not mon.is_atomic)
            surprising_semiknowns = set()
            for mon in remaining_mons:
                if mon not in self.known_moments.keys():
                    value, unknown_factors, known_status = mon.evaluate(
                        atomic_knowns,
                        self.use_lpi_constraints)
                    if known_status == "Known":
                        self.known_moments[mon] = value
                    elif known_status == "Semi":
                        if self.use_lpi_constraints:
                            unknown_mon = \
                                self._monomial_from_atoms(unknown_factors)
                            self.semiknown_moments[mon] = (value, unknown_mon)
                            if self.verbose > 0:
                                if unknown_mon not in self.monomials:
                                    surprising_semiknowns.add(unknown_mon)
                    else:
                        pass
            if (len(surprising_semiknowns) >= 1) and (self.verbose > 0):
                warn("When processing LPI constraints we encountered at " +
                     "least one monomial that does not appear in the " +
                     f"original moment matrix:\n\t{surprising_semiknowns}")
            del atomic_knowns, surprising_semiknowns
        self._cleanup_after_set_values()

    def solve(self,
              interpreter="MOSEKFusion",
              feas_as_optim=False,
              dualise=True,
              solverparameters=None,
              solver_arguments={}):
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
        dualise : bool, optional
            Optimize the dual problem (recommended). By default ``True``.
        solverparameters : dict, optional
            Extra parameters to be sent to the solver. By default ``None``.
        solver_arguments : dict, optional
            By default, solve will use the dictionary of SDP keyword arguments
            given by ``_prepare_solver_arguments()``. However, a user may
            manually override these arguments by passing their own here.
        """
        if not self._relaxation_has_been_generated:
            raise Exception("Relaxation is not generated yet. " +
                            "Call 'InflationSDP.get_relaxation()' first")
        if feas_as_optim and len(self._processed_objective) > 1:
            warn("You have a non-trivial objective, but set to solve a " +
                 "feasibility problem as optimization. Setting "
                 + "feas_as_optim=False and optimizing the objective...")
            feas_as_optim = False

        args = self._prepare_solver_arguments()
        args.update(solver_arguments)
        args.update({"feas_as_optim": feas_as_optim,
                     "verbose": self.verbose,
                     "solverparameters": solverparameters,
                     "solve_dual": dualise})

        self.solution_object = solveSDP_MosekFUSION(**args)

        self.status = self.solution_object["status"]
        if self.status == "feasible":
            self.primal_objective = self.solution_object["primal_value"]
            self.objective_value  = self.solution_object["primal_value"]
            self.objective_value *= (1 if self.maximize else -1)
        else:
            self.primal_objective = self.status
            self.objective_value  = self.status
        collect()

    ###########################################################################
    # PUBLIC ROUTINES RELATED TO THE PROCESSING OF CERTIFICATES               #
    ###########################################################################
    def certificate_as_probs(self,
                             clean: bool = True,
                             chop_tol: float = 1e-10,
                             round_decimals: int = 3) -> sp.core.add.Add:
        """Give certificate as symbolic sum of probabilities. The certificate
        of incompatibility is ``cert < 0``.

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
            The expression of the certificate in terms or probabilities and
            marginals. The certificate of incompatibility is ``cert < 0``.
        """
        try:
            dual = self.solution_object["dual_certificate"]
        except AttributeError:
            raise Exception("For extracting a certificate you need to solve " +
                            "a problem. Call 'InflationSDP.solve()' first.")
        if len(self.semiknown_moments) > 0:
            warn("Beware that, because the problem contains linearized " +
                 "polynomial constraints, the certificate is not guaranteed " +
                 "to apply to other distributions.")
        if clean and not np.allclose(list(dual.values()), 0.):
            dual = clean_coefficients(dual, chop_tol, round_decimals)

        polynomial = sp.S.Zero
        for mon_name, coeff in dual.items():
            if clean and np.isclose(int(coeff), round(coeff, round_decimals)):
                coeff = int(coeff)
            polynomial += coeff * self.names_to_symbols[mon_name]
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
            set to zero. By default ``1e-10``.
        round_decimals : int, optional
            Coefficients that are not set to zero are rounded to the number of
            decimals specified. By default ``3``.

        Returns
        -------
        str
            The certificate in terms of symbols representing the monomials in
            the moment matrix. The certificate of incompatibility is
            ``cert < 0``.
        """
        try:
            dual = self.solution_object["dual_certificate"]
        except AttributeError:
            raise Exception("For extracting a certificate you need to solve " +
                            "a problem. Call 'InflationSDP.solve()' first.")
        if len(self.semiknown_moments) > 0:
            if self.verbose > 0:
                warn("Beware that, because the problem contains linearized " +
                     "polynomial constraints, the certificate is not " +
                     "guaranteed to apply to other distributions.")

        if clean and not np.allclose(list(dual.values()), 0.):
            dual = clean_coefficients(dual, chop_tol, round_decimals)

        rest_of_dual = dual.copy()
        constant_value = rest_of_dual.pop(self.constant_term_name, 0)
        constant_value += rest_of_dual.pop(self.One.name, 0)
        if constant_value:
            if clean:
                cert = "{0:.{prec}f}".format(constant_value,
                                             prec=round_decimals)
            else:
                cert = str(constant_value)
        else:
            cert = ""
        for mon_name, coeff in rest_of_dual.items():
            if mon_name != "0":
                cert += "+" if coeff >= 0 else "-"
                if np.isclose(abs(coeff), 1):
                    cert += mon_name
                else:
                    if clean:
                        cert += "{0:.{prec}f}*{1}".format(abs(coeff),
                                                          mon_name,
                                                          prec=round_decimals)
                    else:
                        cert += f"{abs(coeff)}*{mon_name}"
        cert += " >= 0"
        return cert[1:] if cert[0] == "+" else cert

    ###########################################################################
    # OTHER ROUTINES EXPOSED TO THE USER                                      #
    ###########################################################################
    def build_columns(self,
                      column_specification: Union[str,
                                                  List[List[int]],
                                                  List[sp.core.symbol.Symbol]],
                      max_monomial_length: int = 0):
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
                            columns.append(self.identity_operator)
                    elif type(col) in [sp.core.symbol.Symbol,
                                       sp.core.power.Pow,
                                       sp.core.mul.Mul]:
                        columns.append(self._interpret_name(col))
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
                        lengths = [min(self.inflation_levels[party])
                                   for party in self.hypergraph.T]
                elif len(lengths) == self.nr_parties:
                    lengths = [int(level) for level in lengths]
                else:
                    lengths = [int(lengths)] * self.nr_parties
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
                if spec == "local":
                    col_specs = []
                    for pfreq in party_freqs:
                        operators = []
                        for party in range(self.nr_parties):
                            operators += [party] * pfreq[party]
                        col_specs += [operators]
                    columns = self._build_cols_from_specs(col_specs)
                else:
                    physical_monomials = []
                    for freqs in party_freqs:
                        if freqs == [0] * self.nr_parties:
                            physical_monomials.append(self.identity_operator)
                        else:
                            physmons_per_party = []
                            for party, freq in enumerate(freqs):
                                if freq > 0:
                                    physmons = party_physical_monomials(
                                        self.hypergraph,
                                        self.inflation_levels,
                                        party, freq,
                                        self.setting_cardinalities,
                                        self.outcome_cardinalities,
                                        self._lexorder)
                                    physmons_per_party.append(physmons)
                            for monomial_parts in product(
                                    *physmons_per_party):
                                physical_monomials.append(
                                    self._to_canonical_memoized(
                                        np.concatenate(monomial_parts)))
                    columns = physical_monomials
            else:
                raise Exception("I have not understood the format of the "
                                + "column specification")
        else:
            raise Exception("I have not understood the format of the "
                            + "column specification")

        if not np.array_equal(self._lexorder, self._default_lexorder):
            res_lexrepr = [nb_mon_to_lexrepr(mon, self._lexorder).tolist()
                           if (len(mon) or mon.shape[-1] == 1) else []
                           for mon in columns]
            sorted_mons = sorted(res_lexrepr, key=lambda x: (len(x), x))
            columns = [self._lexorder[lexrepr]
                       if lexrepr != [] else self.identity_operator
                       for lexrepr in sorted_mons]

        columns = [np.array(col,
                            dtype=self.np_dtype).reshape((-1,
                                                          self._nr_properties))
                   for col in columns]
        return columns

    def reset(self, which: Union[str, List[str]]):
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
                self.reset(["bounds", "objective", "values"])
            elif which == "bounds":
                self._reset_bounds()
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

    def write_to_file(self, filename: str):
        """Exports the problem to a file.

        Parameters
        ----------
        filename : str
            Name of the exported file. If no file format is
            specified, it defaults to sparse SDPA format.
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
            print("Writing the SDP program to", filename)
        if extension == "dat-s":
            write_to_sdpa(self, filename)
        elif extension == "csv":
            write_to_csv(self, filename)
        elif extension == "mat":
            write_to_mat(self, filename)
        else:
            raise Exception("File format not supported. Please choose between"
                            + " the extensions .csv, .dat-s and .mat.")

    ###########################################################################
    # ROUTINES RELATED TO CONSTRUCTING COMPOUND MONOMIAL INSTANCES             #
    ###########################################################################
    def _AtomicMonomial(self, array2d: np.ndarray) -> InternalAtomicMonomial:
        """Construct an instance of the `InternalAtomicMonomial` class from
        a 2D array description of a monomial. An atomic moment is a moment that
        does not factorize into a product of other moments. 
        
        A moment `M=<Op1*Op2*...*Opn>` is specified by a 2D array `array2d` with
        `n` rows, one for each operator `Opk`. The order of the rows is the same
        as the order in which the operators are multiplied in the monomial.
        There are `3+nr_sources` columns, one for each property of the operator,
        where `n_sources` is the number of sources in the DAG. The first column
        is an index in `{1,...,nr_parties}` indicating the party, where where
        `nr_parties` is the number of parties in the DAG. The second-to-last and
        last columns encode the setting and the outcome of the operator,
        respectively. The remaining columns in-between, `nr_sources` in number,
        indicate on which copy of the source `i` the operator is acting, with
        value `0` representing no support on the `i`-th source. 
        
        The returned atomic moment is in canonical form, i.e., it cannot be
        brought to a lower lexicographic order through symmetries nor
        commutation rules.  

        Parameters
        ----------
        array2d : np.ndarray
            Operator as an array of integers.

        Returns
        -------
        InternalAtomicMonomial
            An instance of the `InternalAtomicMonomial` class representing the
            inputted moment, brought to canonical form.
            
        Example
        -------
        The following 
        
        >>> InflationSDP._AtomicMonomial(np.array([[1, 0, 1, 2, 2, 3],
                                                   [3, 2, 0, 2, 1, 1]]))
        
        builds the moment `<A^{0,1,2}_{x=2,a=3}*C^{2,0,2}_{z=1,c=1}>`, where
        party `A` acts on copies 1 and 2 of the second and third sources, and
        does not act on the first source, and party `C` acts on copies 2 of the
        first and last sources, and does not act on the second source.
        Furthermore, inflation symmetries are used to bring the moment to
        canonical form, and the final moment returned is
        `<A^{0,1,1}_{x=2,a=3}*C^{1,0,1}_{z=1,c=1}>` which has all operators
        acting on the first source for all sources. Furthermore, the name of
        this atomic monomial is simplified to `'pAC(31|21)'` as this moment can
        be identified with a probability.
        """
        key = self._from_2dndarray(array2d)
        if key in self.atomic_monomial_from_hash:
            return self.atomic_monomial_from_hash[key]
        else:
            repr_array2d = self._to_inflation_repr(array2d)
            new_key      = self._from_2dndarray(repr_array2d)
            if new_key in self.atomic_monomial_from_hash:
                mon = self.atomic_monomial_from_hash[new_key]
                self.atomic_monomial_from_hash[key] = mon
                return mon
            else:
                mon = InternalAtomicMonomial(self, repr_array2d)
                self.atomic_monomial_from_hash[key]     = mon
                self.atomic_monomial_from_hash[new_key] = mon
                return mon

    def Monomial(self, array2d: np.ndarray, idx=-1) -> CompoundMonomial:
        """Create an instance of the `CompoundMonomial` class from a 2D array.

        

        Parameters
        ----------
        array2d : np.ndarray
            _description_
        idx : int, optional
            _description_, by default -1

        Returns
        -------
        CompoundMonomial
            _description_
        """
        _factors = factorize_monomial(array2d, canonical_order=False)
        list_of_atoms = [self._AtomicMonomial(factor)
                         for factor in _factors if len(factor)]
        mon = self._monomial_from_atoms(list_of_atoms)
        mon.attach_idx(idx)
        return mon

    def _conjugate_ndarray(self,
                           mon: np.ndarray,
                           apply_only_commutations=True) -> np.ndarray:
        """
        Compute the canonical form of the conjugate of a monomial.

        Parameters
        ----------
        mon : numpy.ndarray
            Input monomial that cannot be further factorised.
        apply_only_commutations : bool, optional
            If ``True``, skip checking if monomial is zero and if there are
            square projectors.

        Returns
        -------
        numpy.ndarray
            The canonical form of the conjugate of the input monomial under
            relabelling through the inflation symmetries.
        """
        if all_commuting_test(mon, self._lexorder, self._notcomm):
            return mon
        else:
            return self._to_inflation_repr(reverse_mon(mon),
                                           apply_only_commutations)

    def _to_inflation_repr(self,
                           mon: np.ndarray,
                           apply_only_commutations=False) -> np.ndarray:
        r"""Apply inflation symmetries to a monomial in order to bring it to
        its canonical form.

        Example: Assume the monomial is :math:`\langle D^{350}_{00}D^{450}_{00}
        D^{150}_{00}E^{401}_{00}F^{031}_{00}\rangle`. In array form, the
        information about inflation copies is:

        ::

            [[3 5 0],
             [4 5 0],
             [1 5 0],
             [4 0 1],
             [0 3 1]]

        For each column the function assigns to the first row index 1. Then,
        the next different one will be 2, and so on. Therefore, the
        representative of the monomial above is :math:`\langle D^{110}_{00}
        D^{210}_{00} D^{310}_{00} E^{201}_{00} F^{021}_{00} \rangle`.

        Parameters
        ----------
        mon : numpy.ndarray
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
        key = self._from_2dndarray(mon)
        if len(mon) == 0 or np.array_equiv(mon, 0):
            self.canonsym_ndarray_from_hash[key] = mon
            return mon
        else:
            pass
        try:
            return self.canonsym_ndarray_from_hash[key]
        except KeyError:
            pass
        canonical_mon = self._to_canonical_memoized(mon,
                                                    apply_only_commutations)
        canonical_key = self._from_2dndarray(canonical_mon)
        try:
            repr_mon = self.canonsym_ndarray_from_hash[canonical_key]
            self.canonsym_ndarray_from_hash[key] = repr_mon
            return repr_mon
        except KeyError:
            pass
        repr_mon = reduce_inflation_indices(mon)
        repr_key = self._from_2dndarray(repr_mon)
        try:
            real_repr_mon = self.canonsym_ndarray_from_hash[repr_key]
            self.canonsym_ndarray_from_hash[key]           = real_repr_mon
            self.canonsym_ndarray_from_hash[canonical_key] = real_repr_mon
            return real_repr_mon
        except KeyError:
            pass
        other_keys, real_repr_mon = self._inflation_orbit_and_rep(repr_mon)
        other_keys.update({key, canonical_key, repr_key})
        for key in other_keys:
            self.canonsym_ndarray_from_hash[key] = real_repr_mon
        return real_repr_mon

    def _monomial_from_atoms(self,
                             atoms: List[InternalAtomicMonomial]
                             ) -> CompoundMonomial:
        """DOCUMENTATION NEEDED"""
        list_of_atoms = []
        for factor in atoms:
            if factor.is_zero:
                list_of_atoms = [factor]
                break
            elif not factor.is_one:
                list_of_atoms.append(factor)
            else:
                pass
        tuple_of_atoms = tuple(sorted(list_of_atoms))
        try:
            mon = self.monomial_from_atoms[tuple_of_atoms]
            return mon
        except KeyError:
            mon = CompoundMonomial(tuple_of_atoms)
            self.monomial_from_atoms[tuple_of_atoms] = mon
            self.monomial_from_name[mon.name]        = mon
            return mon

    def _inflation_orbit_and_rep(self,
                                 monomial: np.ndarray
                                 ) -> Tuple[set, np.ndarray]:
        """DOCUMENTATION NEEDED"""
        inf_levels = monomial[:, 1:-2].max(axis=0)
        nr_sources = inf_levels.shape[0]
        all_permutations_per_source = [
            format_permutations(list(permutations(range(inflevel))))
            for inflevel in inf_levels.flat]
        seen_hashes = set()
        for permutation in product(*all_permutations_per_source):
            permuted = monomial.copy()
            for source in range(nr_sources):
                permuted = apply_source_perm(permuted,
                                             source,
                                             permutation[source])
            permuted = self._to_canonical_memoized(permuted, True)
            hash     = self._from_2dndarray(permuted)
            seen_hashes.add(hash)
            try:
                representative = self.canonsym_ndarray_from_hash[hash]
                return seen_hashes, representative
            except KeyError:
                pass
        representative = self._to_2dndarray(min(seen_hashes))
        return seen_hashes, representative

    def _sanitise_monomial(self, mon: Any) -> CompoundMonomial:
        """Bring a monomial into the form used internally.
        BETTER DOCUMENTATION NEEDED
        """
        if isinstance(mon, CompoundMonomial):
            return mon
        elif isinstance(mon, (sp.core.symbol.Symbol,
                              sp.core.power.Pow,
                              sp.core.mul.Mul)):
            symbols = flatten_symbolic_powers(mon)
            if len(symbols) == 1:
                try:
                    return self.monomial_from_name[str(symbols[0])]
                except KeyError:
                    pass
            array = np.concatenate([self._interpret_atomic_string(str(op))
                                    for op in symbols])
            return self._sanitise_monomial(array)
        elif isinstance(mon, (tuple, list, np.ndarray)):
            array = np.asarray(mon, dtype=self.np_dtype)
            assert array.ndim == 2, \
                "The monomial representations must be 2d arrays."
            assert array.shape[-1] == self._nr_properties, \
                "The input does not conform to the operator specification."
            canon = self._to_canonical_memoized(array)
            return self.Monomial(canon)
        elif isinstance(mon, str):
            try:
                return self.monomial_from_name[mon]
            except KeyError:
                return self._sanitise_monomial(self._interpret_name(mon))
        elif isinstance(mon, Real):
            if np.isclose(float(mon), 1):
                return self.One
            elif np.isclose(float(mon), 0):
                return self.Zero
            else:
                raise Exception(f"Constant monomial {mon} can only be 0 or 1.")
        else:
            raise Exception(f"sanitise_monomial: {mon} is of type " +
                            f"{type(mon)} and is not supported.")

    ###########################################################################
    # ROUTINES RELATED TO NAME PARSING                                        #
    ###########################################################################
    def _interpret_name(self,
                        monomial: Union[str, sp.core.symbol.Expr, int]
                        ) -> np.ndarray:
        """DOCUMENTATION NEEDED"""
        if isinstance(monomial, sp.core.symbol.Expr):
            factors = [str(factor)
                       for factor in flatten_symbolic_powers(monomial)]
        elif str(monomial) == '1':
            return self.identity_operator
        elif isinstance(monomial, tuple) or isinstance(monomial, list):
            factors = [str(factor) for factor in monomial]
        else:
            assert "^" not in monomial, "Cannot interpret exponents."
            factors = monomial.split("*")
        return np.vstack(tuple(self._interpret_atomic_string(factor_string)
                               for factor_string in factors))

    def _interpret_atomic_string(self, factor_string: str) -> np.ndarray:
        """DOCUMENTATION NEEDED"""
        assert ((factor_string[0] == "<" and factor_string[-1] == ">")
                or set(factor_string).isdisjoint(set("| "))), \
            ("Monomial names must be between < > signs, or in conditional " +
             "probability form.")
        if factor_string[0] == "<":
            operators = factor_string[1:-1].split(" ")
            return np.vstack(tuple(self._interpret_operator_string(op_string)
                                   for op_string in operators))
        else:
            return self._interpret_operator_string(factor_string)[np.newaxis]

    def _interpret_operator_string(self, op_string: str) -> np.ndarray:
        """DOCUMENTATION NEEDED"""
        components = op_string.split("_")
        assert len(components) == self._nr_properties, \
            "Cannot interpret string of this format."
        components[0] = self.names_to_ints[components[0]]
        return np.array([int(s) for s in components], dtype=self.np_dtype)

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
            print("Column structure:", "+".join(to_print))

        columns      = []
        seen_columns = set()
        for block in col_specs:
            if len(block) == 0:
                columns.append(self.identity_operator)
                seen_columns.add(self._from_2dndarray(self.identity_operator))
            else:
                meas_ops = []
                for party in block:
                    meas_ops.append(flatten(self.measurements[party]))
                for monomial_factors in product(*meas_ops):
                    mon   = self._interpret_name(monomial_factors)
                    canon = self._to_canonical_memoized(mon)
                    if not np.array_equal(canon, 0):
                        # If the block is [0, 0], and we have the monomial
                        # A**2 which simplifies to A, then A could be included
                        # in the block [0]. We use the convention that [0, 0]
                        # represents all monomials of length 2 AFTER
                        # simplifications, so we omit monomials of length 1.
                        if canon.shape[0] == len(monomial_factors):
                            key = self._from_2dndarray(canon)
                            if key not in seen_columns:
                                seen_columns.add(key)
                                columns.append(canon)

        return columns

    def _build_momentmatrix(self) -> Tuple[np.ndarray, Dict]:
        """Generate the moment matrix and the correspondence between indices
        and monomials.

        BETTER DOCUMENTATION NEEDED
        """
        problem_arr, canonical_mon_as_bytes_to_idx = \
            calculate_momentmatrix(self.generating_monomials,
                                   self._notcomm,
                                   self._lexorder,
                                   verbose=self.verbose,
                                   commuting=self.commuting,
                                   dtype=self.np_dtype)
        idx_to_canonical_mon = {idx: self._to_2dndarray(mon_as_bytes)
                                for (mon_as_bytes, idx) in
                                canonical_mon_as_bytes_to_idx.items()}
        del canonical_mon_as_bytes_to_idx
        return problem_arr, idx_to_canonical_mon

    def _discover_column_equalities(self) -> List[Tuple[int, Tuple[int]]]:
        """Given the generating monomials, infer implicit equalities between
        columns of the moment matrix. An equality is a dictionary with keys
        being which column and values being coefficients.
        BETTER DOCUMENTATION NEEDED"""
        column_level_equalities = []
        for i, mon in enumerate(self.generating_monomials):
            for k, operator in enumerate(mon):
                party = operator[0] - 1
                # Operators that are involved in normalization equalities are
                # those which are unpacked in non-network scenarios
                if (self.has_children[party]
                   and operator[-1] == self.outcome_cardinalities[party] - 2):
                    operator_2d = np.expand_dims(operator, axis=0)
                    prefix = mon[:k]
                    suffix = mon[(k + 1):]
                    positions = [i]
                    true_cardinality = self.outcome_cardinalities[party] - 1
                    for outcome in range(true_cardinality - 1):
                        variant_operator        = operator_2d.copy()
                        variant_operator[0, -1] = outcome
                        variant_mon             = np.vstack((prefix,
                                                             variant_operator,
                                                             suffix))
                        try:
                            j = self.genmon_hash_to_index[
                                self._from_2dndarray(variant_mon)]
                            positions.append(j)
                        except KeyError:
                            break
                    if len(positions) == true_cardinality:
                        normalization_mon = np.vstack((prefix, suffix))
                        try:
                            j = self.genmon_hash_to_index[
                                self._from_2dndarray(normalization_mon)]
                            column_level_equalities.append((j,
                                                            tuple(positions)))
                        except KeyError:
                            break
        return column_level_equalities

    def _discover_inflation_symmetries(self) -> np.ndarray:
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
        sources_with_copies = [source for source, inf_level
                               in enumerate(self.inflation_levels)
                               if inf_level > 1]
        if len(sources_with_copies):
            inflation_symmetries = []
            identity_perm        = np.arange(self.n_columns, dtype=int)
            for source in tqdm(sources_with_copies,
                               disable=not self.verbose,
                               desc="Calculating symmetries   ",
                               leave=False,
                               position=0):
                one_source_symmetries = [identity_perm]
                inf_level = self.inflation_levels[source]
                perms = format_permutations(list(
                    permutations(range(inf_level)))[1:])
                permutation_failed = False
                for permutation in perms:
                    try:
                        total_perm = np.empty(self.n_columns, dtype=int)
                        for i, mon in enumerate(self.generating_monomials):
                            new_mon = apply_source_perm(mon,
                                                        source,
                                                        permutation)
                            new_mon = self._to_canonical_memoized(new_mon,
                                                                  True)
                            total_perm[i] = self.genmon_hash_to_index[
                                                self._from_2dndarray(new_mon)]
                        one_source_symmetries.append(total_perm)
                    except KeyError:
                        permutation_failed = True
                inflation_symmetries.append(one_source_symmetries)
            if permutation_failed and (self.verbose > 0):
                warn("The generating set is not closed under source swaps."
                     + " Some symmetries will not be implemented.")
            inflation_symmetries = [reduce(np.take, perms) for perms in
                                    product(*inflation_symmetries)]
            return np.unique(inflation_symmetries[1:], axis=0)
        else:
            return np.empty((0, len(self.generating_monomials)), dtype=int)

    def _generate_parties(self) -> List[List[List[List[sp.Symbol]]]]:
        """Generates all the party operators in the quantum inflation.

        Returns
        -------
        List[List[List[List[sympy.Symbol]]]]
            The measurement operators as symbols. The array is indexed as
            measurements[p][c][i][o] for party p, inflation copies c, input i,
            and output o.
        """
        settings = self.setting_cardinalities
        outcomes = self.outcome_cardinalities

        assert len(settings) == len(outcomes), \
            "There\'s a different number of settings and outcomes"
        assert len(settings) == self.hypergraph.shape[1], \
            "The hypergraph does not have as many columns as parties"
        measurements = []
        parties = self.names
        n_states = self.hypergraph.shape[0]
        for pos, [party, ins, outs] in enumerate(zip(parties,
                                                     settings,
                                                     outcomes)):
            party_meas = []
            # Generate all possible copy indices for a party
            all_inflation_indices = product(
                *[list(range(self.inflation_levels[p_idx]))
                  for p_idx in np.nonzero(self.hypergraph[:, pos])[0]])
            # Include zeros in the positions of states not feeding the party
            all_indices = []
            for inflation_indices in all_inflation_indices:
                indices = []
                i = 0
                for idx in range(n_states):
                    if self.hypergraph[idx, pos] == 0:
                        indices.append("0")
                    elif self.hypergraph[idx, pos] == 1:
                        # The +1 is just to begin at 1
                        indices.append(str(inflation_indices[i] + 1))
                        i += 1
                    else:
                        raise Exception("You don\'t have a proper hypergraph")
                all_indices.append(indices)
            # Generate measurements for every combination of indices.
            # The -1 in outs - 1 is because the use of Collins-Gisin notation
            # (see [arXiv:quant-ph/0306129]), whereby the last operator is
            # understood to be written as the identity minus the rest.
            for indices in all_indices:
                meas = generate_operators(
                    [outs - 1 for _ in range(ins)],
                    party + "_" + "_".join(indices)
                )
                party_meas.append(meas)
            measurements.append(party_meas)
        return measurements

    ###########################################################################
    # HELPER FUNCTIONS FOR ENSURING CONSISTENCY                               #
    ###########################################################################
    def _cleanup_after_set_values(self):
        """DOCUMENTATION NEEDED"""
        if self.supports_problem:
            # Convert positive known values into lower bounds.
            nonzero_known_monomials = [mon for
                                       mon, value in self.known_moments.items()
                                       if not np.isclose(value, 0)]
            for mon in nonzero_known_monomials:
                self._processed_moment_lowerbounds[mon] = 1.
                del self.known_moments[mon]
            self.semiknown_moments = dict()

        self._update_lowerbounds()
        self._update_upperbounds()
        self._update_objective()
        num_nontrivial_known = len(self.known_moments)
        if self.momentmatrix_has_a_zero:
            num_nontrivial_known -= 1
        if self.momentmatrix_has_a_one:
            num_nontrivial_known -= 1
        if self.verbose > 1 and num_nontrivial_known > 0:
            print("Number of variables with fixed numeric value:",
                  len(self.known_moments))
        num_semiknown = len(self.semiknown_moments)
        if self.verbose > 1 and num_semiknown > 0:
            print(f"Number of semiknown variables: {num_semiknown}")

    def _reset_bounds(self):
        """Reset the lists of bounds."""
        self._reset_lowerbounds()
        self._reset_upperbounds()
        collect()

    def _reset_lowerbounds(self):
        """Reset the list of lower bounds."""
        self._reset_solution()
        self._processed_moment_lowerbounds = dict()

    def _reset_upperbounds(self):
        """Reset the list of upper bounds."""
        self._reset_solution()
        self._processed_moment_upperbounds = dict()

    def _reset_objective(self):
        """Reset the objective function."""
        self._reset_solution()
        self.objective = {self.One: 0.}
        self._processed_objective = self.objective
        self.maximize = True  # Direction of the optimization

    def _reset_values(self):
        """Reset the known values."""
        self._reset_solution()
        self.known_moments     = dict()
        self.semiknown_moments = dict()
        if self.momentmatrix_has_a_zero:
            self.known_moments[self.Zero] = 0.
        self.known_moments[self.One] = 1.
        collect()

    def _update_objective(self):
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

    def _update_lowerbounds(self):
        """
        Documentation needed.
        """
        for mon, lb in self.moment_lowerbounds.items():
            self._processed_moment_lowerbounds[mon] = \
                max(self._processed_moment_lowerbounds.get(mon, -np.infty), lb)
        for mon, value in self.known_moments.items():
            try:
                lb = self._processed_moment_lowerbounds[mon]
                assert lb <= value, (f"Value {value} assigned for monomial " +
                                     f"{mon} contradicts the assigned lower " +
                                     f"bound of {lb}")
                del self._processed_moment_lowerbounds[mon]
            except KeyError:
                pass

    def _update_upperbounds(self):
        """
        Documentation needed.
        """
        for mon, value in self.known_moments.items():
            try:
                ub = self._processed_moment_upperbounds[mon]
                assert ub >= value, (f"Value {value} assigned for monomial " +
                                     f"{mon} contradicts the assigned upper " +
                                     f"bound of {ub}")
                del self._processed_moment_upperbounds[mon]
            except KeyError:
                pass

    ###########################################################################
    # OTHER ROUTINES                                                          #
    ###########################################################################
    def _atomic_knowable_q(self, atomic_monarray: np.ndarray) -> bool:
        "DOCUMENTATION NEEDED"
        if not is_knowable(atomic_monarray):
            return False
        elif self.network_scenario:
            return True
        else:
            return self._is_knowable_q_non_networks(np.take(atomic_monarray,
                                                            [0, -2, -1],
                                                    axis=1))

    def _dump_to_file(self, filename: str):
        """
        Saves the whole object to a file using `pickle`.

        Parameters
        ----------
        filename : str
            Name of the file.
        """
        import pickle
        with open(filename, "w") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def _from_2dndarray(self, array2d: np.ndarray):
        """Obtains the bytes representation of an array. The library uses this
        representation as hashes for the corresponding monomials.
        """
        return np.asarray(array2d, dtype=self.np_dtype).tobytes()

    def _prepare_solver_arguments(self) -> dict:
        """
        Documentation needed.
        """
        if self.momentmatrix is None:
            raise Exception("Relaxation is not generated yet. " +
                            "Call 'InflationSDP.get_relaxation()' first")

        assert set(self.known_moments.keys()).issubset(self.monomials),\
            ("Error: Tried to assign known values outside of moment matrix: " +
             str(set(self.known_moments.keys()
                     ).difference(self.monomials)))

        solverargs = {"mask_matrices": {mon.name: mon.mask_matrix
                                        for mon in self.monomials},
                      "objective": {mon.name: coeff for mon, coeff
                                    in self._processed_objective.items()},
                      "known_vars": {mon.name: val for mon, val
                                     in self.known_moments.items()},
                      "semiknown_vars": {mon.name: (coeff, subs.name)
                                         for mon, (coeff, subs)
                                         in self.semiknown_moments.items()},
                      "var_equalities": [{mon.name: coeff
                                          for mon, coeff in eq.items()}
                                         for eq in self.moment_equalities],
                      "inequalities": [{mon.name: coeff
                                        for mon, coeff in ineq.items()}
                                       for ineq in self.moment_inequalities]
                      }
        # Add the constant 1 in case of unnormalized problems removed it
        solverargs["known_vars"][self.constant_term_name] = 1.
        for mon, bnd in self._processed_moment_lowerbounds.items():
            lb = {mon.name: 1}
            if not np.isclose(bnd, 0):
                lb[self.constant_term_name] = -bnd
            solverargs["inequalities"].append(lb)
        for mon, bnd in self._processed_moment_upperbounds.items():
            ub = {mon.name: -1}
            if not np.isclose(bnd, 0):
                ub[self.constant_term_name] = bnd
            solverargs["inequalities"].append(ub)
        solverargs["mask_matrices"][self.constant_term_name] = lil_matrix(
            (self.n_columns, self.n_columns))
        return solverargs

    def _reset_solution(self):
        """
        Documentation needed.
        """
        for attribute in {"primal_objective",
                          "objective_value",
                          "solution_object"}:
            try:
                delattr(self, attribute)
            except AttributeError:
                pass
        self.status = "Not yet solved"

    def _set_upperbounds(self, upperbounds: Union[dict, None]):
        """
        Documentation needed.
        """
        self._reset_upperbounds()
        if upperbounds is None:
            return
        sanitized_upperbounds = dict()
        for mon, upperbound in upperbounds.items():
            mon = self._sanitise_monomial(mon)
            if mon not in sanitized_upperbounds.keys():
                sanitized_upperbounds[mon] = upperbound
            else:
                old_bound = sanitized_upperbounds[mon]
                assert np.isclose(old_bound,
                                  upperbound), \
                    (f"Contradiction: Cannot set the same monomial {mon} to " +
                     "have different upper bounds.")
        self._processed_moment_upperbounds = sanitized_upperbounds
        self._update_upperbounds()

    def _set_lowerbounds(self, lowerbounds: Union[dict, None]):
        """
        Documentation needed.
        """
        self._reset_lowerbounds()
        if lowerbounds is None:
            return
        sanitized_lowerbounds = dict()
        for mon, lowerbound in lowerbounds.items():
            mon = self._sanitise_monomial(mon)
            if mon not in sanitized_lowerbounds.keys():
                sanitized_lowerbounds[mon] = lowerbound
            else:
                old_bound = sanitized_lowerbounds[mon]
                assert np.isclose(old_bound, lowerbound), \
                    (f"Contradiction: Cannot set the same monomial {mon} to " +
                     "have different lower bounds.")
        self._processed_moment_lowerbounds = sanitized_lowerbounds
        self._update_lowerbounds()

    def _to_2dndarray(self, bytestream: bytes) -> np.ndarray:
        """Create a monomial array from its corresponding stream of bytes.

        Parameters
        ----------
        bytestream : bytes
            The stream of bytes encoding the monomial.

        Returns
        -------
        numpy.ndarray
            The corresponding monomial in array form.
        """
        array = np.frombuffer(bytestream, dtype=self.np_dtype)
        return array.reshape((-1, self._nr_properties))

    def _to_canonical_memoized(self,
                               array2d: np.ndarray,
                               apply_only_commutations=False) -> np.ndarray:
        """DOCUMENTATION NEEDED"""
        key = self._from_2dndarray(array2d)
        if key in self.canon_ndarray_from_hash:
            return self.canon_ndarray_from_hash[key]
        elif len(array2d) == 0 or np.array_equiv(array2d, 0):
            self.canon_ndarray_from_hash[key] = array2d
            return array2d
        else:
            new_array2d = to_canonical(array2d, self._notcomm, self._lexorder,
                                       self.commuting, apply_only_commutations)
            new_key = self._from_2dndarray(new_array2d)
            self.canon_ndarray_from_hash[key]     = new_array2d
            self.canon_ndarray_from_hash[new_key] = new_array2d
            return new_array2d
