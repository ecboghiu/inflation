"""
The module generates the linear programming relaxation associated to a fanout
or non-fanout inflation instance (see arXiv:1707.06476).

@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""

from collections import Counter, defaultdict
from functools import reduce, cached_property
from gc import collect
from itertools import chain
from numbers import Real
from typing import List, Dict, Tuple, Union, Any
from warnings import warn

import numpy as np
import sympy as sp
from scipy.sparse import coo_array, vstack
from tqdm import tqdm

from .. import InflationProblem
from .lp_utils import solveLP
from .monomial_classes import InternalAtomicMonomial, CompoundMoment
from .numbafied import (nb_outer_bitwise_or,
                        nb_outer_bitwise_xor)
from .writer_utils import write_to_lp, write_to_mps
from ..sdp.fast_npa import nb_is_knowable as is_knowable
from ..sdp.quantum_tools import flatten_symbolic_powers
from ..utils import clean_coefficients, eprint, partsextractor, \
    expand_sparse_vec


class InflationLP(object):
    constant_term_name = "constant_term"

    def __init__(self,
                 inflationproblem: InflationProblem,
                 nonfanout: bool = True,
                 local_level: int = None,
                 supports_problem: bool = False,
                 default_non_negative: bool = True,
                 include_all_outcomes: bool = False,
                 verbose: int = None) -> None:
        """
        Class for generating and solving an LP relaxation for fanout or 
        nonfanout inflation.

        Parameters
        ----------
        inflationproblem : InflationProblem
            Details of the scenario.
        nonfanout : bool, optional
            Whether to consider only nonfanout inflation (GPT problem) or
            fanout inflation (classical problem). By default ``False``.
        local_level : int, optional
            If specified, do not assume nonnegative probability distribution
             over all variables, but only over marginal sets of limited size.
        supports_problem : bool, optional
            Whether to consider feasibility problems with distributions, or 
            just with the distribution's support. By default ``False``.
        default_non_negative : bool, optional
            Whether to set all variables to be non-negative by default. By
            default ``True``.
        include_all_outcomes : bool, optional
            Whether to include all outcomes in the LP, as opposed to using
            Collins-Gisin notation when possible. By default ``False``.
        verbose : int, optional
            Optional parameter for level of verbose:
                * 0: quiet (default),
                * 1: monitor level: track program process and show warnings,
                * 2: debug level: show properties of objects created.
        """
        self.problem_type = "lp"
        self.supports_problem = supports_problem
        if verbose is not None:
            if inflationproblem.verbose > verbose:
                warn("Overriding the verbosity from InflationProblem")
            self.verbose = verbose
        else:
            self.verbose = inflationproblem.verbose
        self.local_level = local_level
        if not nonfanout:
            assert not inflationproblem._default_notcomm.any(), \
                "You appear to be requesting fanout (classical)" \
                    + " inflation, \nbut have not specified classical_sources=`all`." \
                    + "\nNote that the `nonfanout` keyword argument is deprecated as of release 2.0.0"

        self.all_operators_commute = True
        self.all_commuting_q_2d = lambda mon: True
        self.all_commuting_q_1d = lambda lexmon: True

        self.default_non_negative = default_non_negative
        if self.verbose > 1:
            print(inflationproblem)

        # Inheritance
        self.InflationProblem = inflationproblem
        self.names = inflationproblem.names
        self.names_to_ints = inflationproblem.names_to_ints
        self._lexrepr_to_names = inflationproblem._lexrepr_to_names
        self._lexrepr_to_copy_index_free_names = inflationproblem._lexrepr_to_copy_index_free_names
        self.op_from_name = dict()
        for i, op_names in enumerate(inflationproblem._lexrepr_to_all_names.tolist()):
            for op_name in op_names:
                self.op_from_name.setdefault(op_name, i)
        self.nr_sources = inflationproblem.nr_sources
        self.nr_parties = inflationproblem.nr_parties
        self.hypergraph = inflationproblem.hypergraph
        self.network_scenario = inflationproblem.is_network
        self.inflation_levels = inflationproblem.inflation_level_per_source
        self.setting_cardinalities = inflationproblem.settings_per_party
        self.outcome_cardinalities = inflationproblem.outcomes_per_party
        self.private_setting_cardinalities = inflationproblem.private_settings_per_party
        self.expected_distro_shape = inflationproblem.expected_distro_shape
        self.rectify_fake_setting = inflationproblem.rectify_fake_setting
        self.factorize_monomial_1d = inflationproblem.factorize_monomial_1d
        self._is_knowable_q_non_networks = \
            inflationproblem._is_knowable_q_non_networks
        self._nr_properties = inflationproblem._nr_properties
        self.np_dtype = inflationproblem._np_dtype
        self._lexorder = inflationproblem._lexorder
        self._nr_operators = inflationproblem._nr_operators
        self._lexrange = np.arange(self._nr_operators)
        self.lexorder_symmetries = inflationproblem.lexorder_symmetries
        self._from_2dndarray = inflationproblem._from_2dndarray
        self.mon_to_lexrepr = inflationproblem.mon_to_lexrepr
        self.blank_bool_vec = np.zeros(self._nr_operators, dtype=bool)
        self._ortho_groups_per_party = inflationproblem._ortho_groups_per_party
        self.has_children = inflationproblem.has_children.copy()
        if include_all_outcomes or supports_problem: # HACK to fix detection of incompatible supports. (Can be fixed upon adding set_extra_equalities)
            self.has_children[:] = True
        self.does_not_have_children = np.logical_not(self.has_children)

        all_ortho_groups_as_boolarrays = []
        for ortho_groups in self._ortho_groups_per_party:
            ortho_groups_as_boolarrays = []
            for ortho_group in ortho_groups:
                ortho_groups_as_boolarrays.append(np.vstack(
                    [self.mon_to_boolvec(op[np.newaxis])
                     for op in ortho_group]))
            all_ortho_groups_as_boolarrays.append(ortho_groups_as_boolarrays)
        CG_adjusting_ortho_groups_as_boolarrays = []
        CG_nonadjusting_ortho_groups_as_boolarrays = []
        for i, ortho_groups_as_boolarrays in enumerate(all_ortho_groups_as_boolarrays):
            if self.does_not_have_children[i]:
                CG_adjusting_ortho_groups_as_boolarrays.append(ortho_groups_as_boolarrays)
            else:
                CG_nonadjusting_ortho_groups_as_boolarrays.append(
                    ortho_groups_as_boolarrays)

        self._CG_adjusting_ortho_groups_as_boolarrays = list(chain.from_iterable(CG_adjusting_ortho_groups_as_boolarrays))
        self._CG_nonadjusting_ortho_groups_as_boolarrays = list(chain.from_iterable(CG_nonadjusting_ortho_groups_as_boolarrays))
        self._all_ortho_groups_as_boolarrays = list(chain.from_iterable(all_ortho_groups_as_boolarrays))

        # We want to consider ALL outcomes for variables which have children, 
        # but not the last outcome for childless variables.
        # In other words, if we take all the CG equalities, we want to convert 
        # them to inequalities when the LHS is non_CG but childless-only.
        if np.any(self.does_not_have_children):
            bad_boolvecs_for_ineqs = [bool_array[0] for bool_array in self._CG_adjusting_ortho_groups_as_boolarrays]
            self._boolvec_for_CG_ineqs = np.bitwise_or.reduce(bad_boolvecs_for_ineqs, axis=0)
        else:
            self._boolvec_for_CG_ineqs = self.blank_bool_vec
        if np.any(self.has_children):
            bad_boolvecs_for_eqs = [bool_array[0] for bool_array in self._CG_nonadjusting_ortho_groups_as_boolarrays]
            self._boolvec_for_FR_eqs = np.bitwise_or.reduce(bad_boolvecs_for_eqs, axis=0)
        else:
            self._boolvec_for_FR_eqs = self.blank_bool_vec

        if self.verbose > 1:
            print("Number of single operator measurements per party:", end="")
            prefix = " "
            for i, measures in enumerate(inflationproblem.measurements):
                op_count = np.prod(measures.shape[:2])
                print(prefix + f"{self.names[i]}={op_count}", end="")
                prefix = ", "
            print()
        self.use_lpi_constraints = False

        self.identity_operator = np.empty((0, self._nr_properties),
                                          dtype=self.np_dtype)
        self.zero_operator = np.zeros((1, self._nr_properties),
                                      dtype=self.np_dtype)

        # These next properties are reset during generate_lp, but are needed in
        # init so as to be able to test the Monomial constructor function
        # without generate_lp.

        self._atomic_monomial_from_hash  = dict()
        self.monomial_from_atoms        = dict()
        self.monomial_from_name = dict()
        self.monomial_from_symbol = dict()
        self.One  = self.Monomial(self.identity_operator, idx=1)
        self._generate_lp()

    ###########################################################################
    # MAIN ROUTINES EXPOSED TO THE USER                                       #
    ###########################################################################
    @cached_property
    def monomials_as_strings(self):
        """Returns the monomials as strings."""
        return [mon.name for mon in self.monomials]

    def set_bounds(self,
                   bounds: Union[dict, None],
                   bound_type: str = "up") -> None:
        r"""Set numerical lower or upper bounds on the moments generated in the
        LP relaxation. The bounds are at the level of the LP variables,
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
        >>> self.set_bounds({"pAB(00|00)": 0.2}, "lo")
        """
        assert bound_type in ["up", "lo"], \
            "The 'bound_type' argument should be either 'up' or 'lo'"
        if bound_type == "up":
            self._set_upperbounds(bounds)
        else:
            self._set_lowerbounds(bounds)

    @cached_property
    def atomic_factors(self):
        atoms = set()
        for mon in self.monomials:
            atoms.update(mon.factors)
        return sorted(atoms)

    @cached_property
    def atomic_monomials(self):
        """Returns the atomic monomials."""
        return [self._monomial_from_atoms([atom]) for atom in
                self.atomic_factors]

    @cached_property
    def atom_from_name(self):
        """Returns the atomic monomials."""
        lookup_dict = dict()
        for atom in self.atomic_factors:
            lookup_dict[atom.name] = atom
            lookup_dict[atom.legacy_name] = atom
        return lookup_dict

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
    def factorization_conditions(self):
        """Returns the factorization conditions."""
        conds = dict()
        for mon in self.monomials:
            if mon.n_factors > 1:
                conds[mon] = tuple(self.monomial_from_atoms[(fac,)]
                                   for fac in mon.factors)
        return conds

    @cached_property
    def quadratic_factorization_conditions(self):
        """Returns the quadratic factorization conditions."""
        conds = dict()
        for mon in self.monomials:
            if mon.n_factors > 1:
                conds[mon] = (self.monomial_from_atoms[mon.factors[:1]],
                              self.monomial_from_atoms[mon.factors[1:]])
        return conds

    def set_distribution(self,
                         prob_array: Union[np.ndarray, None],
                         use_lpi_constraints: bool = False,
                         shared_randomness: bool = False) -> None:
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
            ``InflationProblem`` used to instantiate ``InflationLP``.
        use_lpi_constraints : bool, optional
            Specification whether linearized polynomial constraints (see,
            e.g., Eq. (D6) in `arXiv:2203.16543
            <https://www.arxiv.org/abs/2203.16543/>`_) will be imposed or
            not. By default ``False``.
        shared_randomness : bool, optional
            Specification whether higher order monomials may be calculated.
            If universal shared randomness is present (i.e., the flag is
            ``True``), only atomic monomials are assigned numerical values.
        """
        if prob_array is not None:
            assert prob_array.shape == self.expected_distro_shape, f"Cardinalities mismatch: \n" \
                                                                   f"expected {self.expected_distro_shape}, \n " \
                                                                   f"got {prob_array.shape}"
            knowable_values = {atom: atom.compute_marginal(prob_array)
                               for atom in self.knowable_atoms}
        else:
            knowable_values = dict()

        self.set_values(knowable_values,
                        use_lpi_constraints=use_lpi_constraints,
                        only_specified_values=shared_randomness)

    def set_objective(self,
                      objective: Union[sp.core.expr.Expr, dict, None],
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
        else:
            if self.use_lpi_constraints and self.verbose > 0:
                warn("You have the flag `use_lpi_constraints` set to True. Be "
                     + "aware that imposing linearized polynomial constraints "
                     + "will constrain the optimization to distributions with "
                     + "fixed marginals.")
            sign = (1 if self.maximize else -1)
            self.objective = {mon: (sign * coeff)
                              for mon, coeff in self._sanitise_dict(objective).items()}
            surprising_objective_terms = {mon for mon in self.objective.keys()
                                          if mon not in self.monomials}
            assert len(surprising_objective_terms) == 0, \
                ("When interpreting the objective we have encountered at " +
                 "least one monomial that does not appear in the original " +
                 f"generating set:\n\t{surprising_objective_terms}")

    def update_values(self,
                      values: Union[Dict[Union[CompoundMoment,
                      InternalAtomicMonomial,
                      sp.core.symbol.Symbol,
                      str],
                      Union[float, sp.core.expr.Expr]],
                      None],
                      use_lpi_constraints: bool = False,
                      only_specified_values: bool = False) -> None:
        """Directly assign numerical values to variables in the generating set.
        This is done via a dictionary where keys are the variables to have
        numerical values assigned (either in their operator form, in string
        form, or directly referring to the variable in the generating set), and
        the values are the corresponding numerical quantities.

        Parameters
        ----------
        values : Union[None, Dict[Union[CompoundMoment, InternalAtomicMonomial, sympy.core.symbol.Symbol, str], float]]
            The description of the variables to be assigned numerical values
            and the corresponding values. The keys can be either of the
            Monomial class, symbols or strings (which should be the name of
            some Monomial).
        use_lpi_constraints : bool
            Specification whether linearized polynomial constraints (see, e.g.,
            Eq. (D6) in `arXiv:2203.16543
            <https://www.arxiv.org/abs/2203.16543/>`_) will be imposed or not.
            By default ``False``.
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
        for mon, value in values.items():
            try:  # We do this to deal with NaNs (such as in undefined supports)
                  # in a way that is also compatible with symbolic values.
                if np.isnan(value):
                    continue
            except TypeError:
                pass
            mon = self._sanitise_monomial(mon)
            self.known_moments[mon] = value
        if not only_specified_values:
            atomic_knowns = {mon.factors[0]: val
                             for mon, val in self.known_moments.items()
                             if len(mon) == 1}
            monomials_not_present = set(self.known_moments.keys()
                                        ).difference(self.monomials)
            for mon in monomials_not_present:
                warn(f"We do not recognize the set value of {mon} "
                     + "as it does not appear in the internal list of monomials.")
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
            for mon in tqdm(list(remaining_mons), disable=not self.verbose,
                                 desc="Evaluating monomials    "):
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
                                raise Exception(
                                    "When processing LPI constraints we encountered a " +
                                    "monomial that does not appear in the list of LP " +
                                    f"variables:\n\t{unknown_mon}")
                else:
                    pass
            if (len(surprising_semiknowns) >= 1) and (self.verbose > 0):
                warn("When processing LPI constraints we encountered at " +
                     "least one monomial that does not appear in the " +
                     f"generating set:\n\t{surprising_semiknowns}")
            del atomic_knowns, surprising_semiknowns
        self._cleanup_after_set_values()

    def set_values(self, 
                   values: Union[Dict[Union[CompoundMoment,
                   InternalAtomicMonomial,
                   sp.core.symbol.Symbol,
                   str],
                   Union[float, sp.core.expr.Expr]],
                   None], 
                   **kwargs):
        r"""Exactly like update_values, except it resets all known values to
        zero as an intermediate step.
        
        Parameters
        ----------
        values : Union[None, Dict[Union[CompoundMoment, InternalAtomicMonomial, sympy.core.symbol.Symbol, str], float]]
            The description of the variables to be assigned numerical values
            and the corresponding values. The keys can be either of the
            Monomial class, symbols or strings (which should be the name of
            some Monomial).
        """
        self._reset_values()
        if (values is None) or len(values) == 0:
            self._cleanup_after_set_values()
            return
        else:
            self.update_values(values, **kwargs)
            return

    def set_extra_equalities(self,
                             extra_equalities: Union[list, None]) -> None:
        """Set extra equality constraints for the LP.

        Parameters
        ----------
        extra_equalities : Union[list, None]
            List of dictionaries representing additional equality constraints.
            The keys (variables) can be strings, instances of
            `CompoundMonomial`, or monomials as 2D arrays.
        """
        self.extra_equalities = []  # Reset every time
        if not extra_equalities or extra_equalities is None:
            return
        self.extra_equalities = [self._sanitise_dict(eq)
                                 for eq in extra_equalities]

    def set_extra_inequalities(self,
                               extra_inequalities: Union[list, None]) -> None:
        """Set extra inequality constraints for the LP.

        Parameters
        ----------
        extra_inequalities : Union[list, None]
            List of dictionaries representing additional inequality
             constraints. The keys (variables) can be strings, instances of
            `CompoundMonomial`, or monomials as 2D arrays.
        """
        self.extra_inequalities = []  # Reset every time
        if not extra_inequalities or extra_inequalities is None:
            return
        self.extra_inequalities = [self._sanitise_dict(ineq)
                                   for ineq in extra_inequalities]

    def solve(self,
              relax_known_vars: bool = False,
              relax_inequalities: bool = False,
              solve_dual: bool = True,
              solverparameters: dict = None,
              verbose: int = None,
              default_non_negative: bool = None,
              **solver_arguments) -> None:
        r"""Call a solver on the LP relaxation. Upon successful solution, it
        returns the primal and dual objective values along with the solution
        matrices.

        Parameters
        ----------
        relax_known_vars : bool, optional
            Do feasibility as optimization where each known value equality
            becomes two relaxed inequality constraints. E.g., P(A) = 0.7
            becomes P(A) + lambda >= 0.7 and P(A) - lambda <= 0.7, where lambda
            is a slack variable. By default, ``False``.
        relax_inequalities : bool, optional
            Do feasibility as optimization where each inequality is relaxed by
            the non-negative slack variable lambda. By default, ``False``.
        solve_dual : bool, optional
            Optimize the dual problem (recommended). By default ``False``.
        solverparameters : dict, optional
            Extra parameters to be sent to the solver. By default ``None``.
        verbose : int, optional
            Verbosity level.
        default_non_negative : bool, optional
            If ``True``, all variables are set to be non-negative by default.
        solver_arguments : dict, optional
            By default, solve will use the dictionary of LP keyword arguments
            given by ``_prepare_solver_arguments()``. However, a user may
            manually override these arguments by passing their own here.
        """
        if (relax_known_vars or relax_inequalities) and \
                len(self.objective) > 0:
            warn("You have a non-trivial objective, but set to solve a "
                 "feasibility problem as optimization. Setting "
                 "relax_known_vars=False, relax_inequalities=False, and "
                 "optimizing the objective...")
            relax_known_vars = relax_inequalities = False
        if verbose is None:
            real_verbose = self.verbose
        else:
            real_verbose = verbose
        if default_non_negative is None:
            real_default_non_negative = self.default_non_negative
        else:
            real_default_non_negative = default_non_negative

        args = self._prepare_solver_matrices()
        
        # Still allow for 'feas_as_optim' as an argument
        if 'feas_as_optim' in solver_arguments:
            if solver_arguments["feas_as_optim"]:
                relax_known_vars = False
                relax_inequalities = True
                del solver_arguments["feas_as_optim"]
            else:
                relax_known_vars = relax_inequalities = False
                del solver_arguments["feas_as_optim"]
        args.update(solver_arguments)

        args.update({"relax_known_vars": relax_known_vars,
                     "relax_inequalities": relax_inequalities,
                     "verbose": real_verbose,
                     "default_non_negative": real_default_non_negative,
                     "solverparameters": solverparameters,
                     "solve_dual": solve_dual})

        self.solution_object = solveLP(**args)
        self.success = self.solution_object["success"]
        self.status = self.solution_object["status"]
        if self.success:
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
                            "a problem. Call \"InflationLP.solve()\" first.")
        if len(self.semiknown_moments) > 0:
            warn("Beware that, because the problem contains linearized " +
                 "polynomial constraints, the certificate is not guaranteed " +
                 "to apply to other distributions.")
        if np.allclose(list(dual.values()), 0.):
            return dict()
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
                         dict_with_monomial_keys) -> str:
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
        if not len(polynomial_as_str):  # Failsafe for empty dictionary.
            polynomial_as_str = "0"
        elif polynomial_as_str[0] == "+":
            polynomial_as_str = polynomial_as_str[1:]
        return polynomial_as_str

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

    ###########################################################################
    # OTHER ROUTINES EXPOSED TO THE USER                                      #
    ##########################################################################
    def reset(self, which: Union[str, List[str]] = "all") -> None:
        """Reset the various user-specifiable objects in the inflation LP.

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
                                "InflationLP.")
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
            defaults to the LP format. Supported formats are ``.lp`` (LP) and
            ``.mps`` (MPS).
        """
        # Determine file extension
        parts = filename.split(".")
        if len(parts) >= 2:
            extension = parts[-1]
        else:
            extension = "lp"
            filename += ".lp"

        # Write file according to the extension
        args = self._prepare_solver_arguments(separate_bounds=True)
        if self.verbose > 0:
            print("Writing the LP program to", filename)
        if extension == "lp":
            write_to_lp(args, filename)
        elif extension == "mps":
            write_to_mps(args, filename)
        else:
            raise Exception("File format not supported. Please choose between "
                            + "the extensions `.lp` and `.mps`.")

    ###########################################################################
    # ROUTINES RELATED TO CONSTRUCTING COMPOUND MONOMIAL INSTANCES            #
    ###########################################################################
    def _AtomicMonomial(self,
                        array1d: np.ndarray) -> InternalAtomicMonomial:
        """Construct an instance of the `InternalAtomicMonomial` class from
        a 2D array description of a monomial.

        See the documentation of the `InternalAtomicMonomial` class for more
        details.

        Parameters
        ----------
        array1d : numpy.ndarray
            Monomial encoded as a 1D array of integers relative to _lexorder.

        Returns
        -------
        InternalAtomicMonomial
            An instance of the `InternalAtomicMonomial` class representing the
            input.
        """
        key = self.blank_bool_vec.copy()  # Quantum case will be different
        key[array1d] = True
        try:
            return self._atomic_monomial_from_hash[key.tobytes()]
        except KeyError:
            if len(self.lexorder_symmetries) == 1:
                mon = InternalAtomicMonomial(self, array1d)
                self._atomic_monomial_from_hash[key.tobytes()] = mon
                return mon
            else:
                mon_as_symboolvec = key[self.lexorder_symmetries]
                mon_as_symboolvec = mon_as_symboolvec[
                    np.lexsort(mon_as_symboolvec.T[::-1])]
                mon_as_boolvec = mon_as_symboolvec[-1]
                mon = InternalAtomicMonomial(self, 
                                             np.flatnonzero(mon_as_boolvec))
                for alt_key in mon_as_symboolvec:
                    self._atomic_monomial_from_hash[alt_key.tobytes()] = mon
                return mon

    def Monomial(self, array1d: np.ndarray, idx: int = -1) -> CompoundMoment:
        r"""Create an instance of the `CompoundMonomial` class from a 2D array.
        An instance of `CompoundMonomial` is a collection of
        `InternalAtomicMonomial`.

        Parameters
        ----------
        array1d : numpy.ndarray
            Moment encoded as a 1D array of integers, relative to _lexorder
        idx : int, optional
            Assigns an integer index to the resulting monomial, which can be
            used as an id, by default -1.

        Returns
        -------
        CompoundMoment
            The monomial factorised into AtomicMonomials, all brought to
            representative form under inflation symmetries.
        """
        _factors = self.factorize_monomial_1d(array1d, canonical_order=False)
        list_of_atoms = [self._AtomicMonomial(factor)
                         for factor in _factors if len(factor)]
        mon = self._monomial_from_atoms(list_of_atoms)
        mon.attach_idx(idx)
        return mon

    def _monomial_from_atoms(self,
                             atoms: List[InternalAtomicMonomial]
                             ) -> CompoundMoment:
        """Build an instance of `CompoundMonomial` from a list of instances
        of `InternalAtomicMonomial`.

        Parameters
        ----------
        atoms : List[InternalAtomicMonomial]
            List of instances of `InternalAtomicMonomial`.

        Returns
        -------
        CompoundMoment
            A `CompoundMonomial` with atomic factors given by `atoms`.
        """
        key = tuple(sorted(atoms))
        try:
            return self.monomial_from_atoms[key]
        except KeyError:
            mon = CompoundMoment(atoms)
            try:
                mon.idx = self.first_free_idx
                self.first_free_idx += 1
            except AttributeError:
                pass
            self.monomial_from_atoms[key] = mon
            self.monomial_from_name[mon.name] = mon
            self.monomial_from_name[mon.legacy_name] = mon  # For legacy compatibility!
            self.monomial_from_symbol[mon.symbol] = mon
            return mon

    def _sanitise_monomial(self, mon: Any) -> CompoundMoment:
        """Return a ``CompoundMonomial`` built from ``mon``, where ``mon`` can
        be either the name of a moment as a string, a SymPy variable, a
        monomial encoded as a 2D array, or an integer in case the moment is the
        unit moment or the zero moment.


        Parameters
        ----------
        mon : Any
            The name of a moment as a string, a SymPy variable with the name of
            a valid moment, a 2D array encoding of a moment or an integer in
            case the moment is the unit moment or the zero moment.

        Returns
        -------
        CompoundMoment
            Instance of ``CompoundMonomial`` built from ``mon``.

        Raises
        ------
        Exception
            If ``mon`` is the constant monomial, it can only be numbers 0 or 1
        Exception
            If the type of ``mon`` is not supported.
        """
        if isinstance(mon, CompoundMoment):
            return mon
        elif isinstance(mon, InternalAtomicMonomial):
            return self._monomial_from_atoms([mon])
        elif isinstance(mon, (sp.core.symbol.Symbol,
                              sp.core.power.Pow,
                              sp.core.mul.Mul,
                              sp.core.symbol.Expr)):
            try:
                return self.monomial_from_symbol[mon]
            except KeyError:
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
            assert array.ndim <= 2, \
                "The monomial representations must be 1d or 2d arrays."
            if array.ndim == 2:
                return self._sanitise_monomial(self.mon_to_lexrepr(array))
            elif array.ndim == 1:
                return self.Monomial(array)
            else:
                assert array.ndim == 2, \
                    "The monomial representations must be 2d arrays."
        elif isinstance(mon, str):
            try:
                return self.monomial_from_name[mon]
            except KeyError:
                print(f"As of now we only recognize \n{list(self.monomial_from_name.keys())}")
                return self._sanitise_monomial(self._interpret_name(mon))
        elif isinstance(mon, Real):
            if np.isclose(float(mon), 1):
                return self.One
            else:
                raise Exception(f"Constant monomial {mon} can only be 1.")
        else:
            raise Exception(f"sanitise_monomial: {mon} is of type " +
                            f"{type(mon)} and is not supported.")

    def _sanitise_dict(self, input_dict: Any) -> Dict:
        """Sanitize a dictionary of monomials.

        Parameters
        ----------
        input_dict : Any
            The dictionary to be sanitized.

        Returns
        -------
        Dict
            The sanitized dictionary.
        """
        if isinstance(input_dict, sp.core.expr.Expr):
            if input_dict.free_symbols:
                input_dict_copy = {k: float(v) for k, v in sp.expand(input_dict).as_coefficients_dict().items()}
            else:
                input_dict_copy = dict()
        else:
            input_dict_copy = input_dict
        output_dict = defaultdict(int)
        for k, v in input_dict_copy.items():
            if not np.isclose(v, 0):
                output_dict[self._sanitise_monomial(k)] += v
        return output_dict

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
        elif isinstance(monomial, tuple) or isinstance(monomial, list):
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
                cleaned_factor_string = cleaned_factor_string.replace(substring, '')
            cleaned_factor_string = cleaned_factor_string.replace( ' & ',' ')
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
    # ROUTINES RELATED TO THE GENERATION OF THE LP                            #
    ###########################################################################
    def _generate_lp(self) -> None:
        """Creates the LP associated with the inflation problem.

        In the inflated graph there are many symmetries coming from invariance
        under swaps of the copied sources, which are used to remove variables
        from the LP.
        """
        # Note that there IS NO POSSIBILITY OF ORTHOGONALITY in the LP
        # and hence we start indices from zero, always.
        try:
            self.generate_lp_has_been_called += 1
        except AttributeError:
            self.generate_lp_has_been_called = 0
        if self.generate_lp_has_been_called:
            return None

        self._atomic_monomial_from_hash = dict()
        self.monomial_from_atoms = dict()
        self.Constant_Term = self.One.__copy__()
        self.Constant_Term.name = self.constant_term_name
        self.monomial_from_name[self.constant_term_name] = self.Constant_Term

        (self._raw_monomials_as_lexboolvecs,
         self._raw_monomials_as_lexboolvecs_non_CG) = self._build_raw_lexboolvecs()
        collect(generation=2)
        self.raw_n_columns = len(self._raw_monomials_as_lexboolvecs)
        self.raw_n_columns_non_CG = len(self._raw_monomials_as_lexboolvecs_non_CG)

        self._raw_lookup_dict = {bitvec.tobytes(): i for i, bitvec in
                                 enumerate(self._raw_monomials_as_lexboolvecs)}

        symmetrization_required = np.any(self.inflation_levels - 1)
        if symmetrization_required:
            # Calculate the inflation symmetries
            if self.verbose > 0:
                eprint("Initiating symmetry calculation...")
            orbits = self._discover_inflation_orbits(self._raw_monomials_as_lexboolvecs,
                                                     raw_hash_table=self._raw_lookup_dict)
            if self.verbose > 1:
                eprint("Halfway through symmetry calculations...")
            old_reps_CG, unique_indices_CG, inverse_CG = np.unique(
                orbits,
                return_index=True,
                return_inverse=True)
            self.num_CG = len(old_reps_CG)

            orbits_non_CG = self._discover_inflation_orbits(
                self._raw_monomials_as_lexboolvecs_non_CG)
            old_reps_non_CG, unique_indices_non_CG, inverse_non_CG = np.unique(
                orbits_non_CG, return_index=True, return_inverse=True)
            self.num_non_CG = len(old_reps_non_CG)
            if self.verbose > 1:
                print(f"Orbits discovered! {self.num_CG} unique monomials.")
            # Obtain the real generating monomials after accounting for symmetry
        else:
            self.num_CG = self.raw_n_columns
            unique_indices_CG = np.arange(self.num_CG)
            inverse_CG = unique_indices_CG
            self.num_non_CG = self.raw_n_columns_non_CG
            unique_indices_non_CG = np.arange(self.num_non_CG)
        self.inverse = inverse_CG

        self._monomials_as_lexboolvecs = self._raw_monomials_as_lexboolvecs[unique_indices_CG]
        self._monomials_as_lexboolvecs_non_CG = self._raw_monomials_as_lexboolvecs_non_CG[unique_indices_non_CG]
        self.n_columns = len(self._monomials_as_lexboolvecs)

        self.nof_collins_gisin_inequalities = self.num_non_CG

        if self.verbose > 0:
            eprint("Number of variables in the LP:",
                  self.n_columns)
            eprint("Number of nontrivial inequality constraints in the LP:",
                    self.nof_collins_gisin_inequalities)

        _monomials_as_lexorder = [tuple(self.mon_to_lexrepr(self._lexorder[bool_idx]))
                                           for bool_idx in
                                           self._monomials_as_lexboolvecs]

        # Associate Monomials to the remaining entries.
        _compmonomial_to_idx = dict()
        self.extra_inverse = np.arange(self.n_columns, dtype=int)
        first_free_index = 0
        for idx, mon_as_lexboolvec in tqdm(enumerate(self._monomials_as_lexboolvecs),
                             disable=not self.verbose,
                             desc="Initializing monomials   ",
                             total=self.n_columns):
            mon = self.Monomial(np.flatnonzero(mon_as_lexboolvec), first_free_index)
            try:
                current_index = _compmonomial_to_idx[mon]
                mon.idx = current_index
            except KeyError:
                current_index = first_free_index
                _compmonomial_to_idx[mon] = current_index
                first_free_index += 1
            self.extra_inverse[idx] = current_index
        self.inverse = self.extra_inverse[self.inverse] # Hack to allow for powerful symmetries

        monomials_as_list = list(_compmonomial_to_idx.keys())
        self.monomials = np.array(monomials_as_list, dtype = object)
        # assert np.array_equal(list(_compmonomial_to_idx.values()), np.arange(len(self.monomials))), "Something went wrong with monomial initialization."
        old_num_columns = self.n_columns
        self.n_columns = len(self.monomials)
        self.first_free_idx = first_free_index
        self.monomial_names = np.array([mon.name for mon in monomials_as_list])
        if self.n_columns < old_num_columns:
            if self.verbose > 0:
                eprint("Further variable reduction has been made possible. Number of variables in the LP:",
                       self.n_columns)
        self.compmonomial_from_idx = dict(zip(range(self.n_columns), monomials_as_list))
        self.compmonomial_to_idx = dict(zip(monomials_as_list, range(self.n_columns)))
        del _compmonomial_to_idx, monomials_as_list
        collect(generation=2)
        assert self.monomials[0] == self.One, "Sparse indexing requires that first column represent one."

        # assert len(self.compmonomial_to_idx.keys()) == self.n_columns, \
        #     (f"Multiple indices are being associated to the same monomial. \n" +
        #     f"Expected {self.n_columns}, got {len(self.compmonomial_to_idx.keys())}.")


        if self.verbose > 1:
            _counter = Counter([mon.knowability_status
                                for mon in self.monomials])
            self.n_knowable           = _counter["Knowable"]
            self.n_something_knowable = _counter["Semi"]
            self.n_unknowable         = _counter["Unknowable"]
            eprint(f"The problem has {self.n_knowable} knowable monomials, " +
                  f"{self.n_something_knowable} semi-knowable monomials, " +
                  f"and {self.n_unknowable} unknowable monomials.")

        # This dictionary useful for certificates_as_probs
        self.names_to_symbols = {mon.name: mon.symbol
                                 for mon in self.monomials}
        self.names_to_symbols[self.constant_term_name] = sp.S.One

        self._set_lowerbounds(None)
        self._set_upperbounds(None)
        self.set_objective(None)
        self.set_values(None)

        self._lp_has_been_generated = True
        if self.verbose > 1:
            print("LP initialization complete, ready to accept further specifics.")

    def _build_raw_lexboolvecs(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""Creates the generating set of monomials (as boolvecs),
        both in and out of Collins-Gisin notation.
        """
        choices_to_combine = []
        lengths = []
        for party in range(self.nr_parties):
            boolvecs = \
                self.InflationProblem._generate_compatible_monomials_given_party(
                    party,
                    up_to_length=self.local_level,
                    with_last_outcome=True)
            lengths.append(len(boolvecs))
            choices_to_combine.append(boolvecs)
        # Use reduce to take outer combinations, using bitwise addition
        if self.verbose > 0:
            eprint(f"About to generate {np.prod(np.asarray(lengths, dtype=object))} probability placeholders...")

        choices_to_combine_CG = []
        choices_to_combine_global = []
        for choices in choices_to_combine:
            CG_selection = np.logical_not(np.matmul(choices,
                                                    self._boolvec_for_CG_ineqs))
            choices_to_combine_CG.append(choices[CG_selection])
            event_count = choices.sum(axis=1)
            choices_to_combine_global.append(choices[event_count == event_count.max()])
        raw_boolvecs_CG = reduce(nb_outer_bitwise_or, 
                                 reversed(choices_to_combine_CG))
        raw_boolvecs_global = reduce(nb_outer_bitwise_or,
                                     reversed(choices_to_combine_global))
        return (raw_boolvecs_CG, raw_boolvecs_global)

    @cached_property
    def minimal_sparse_equalities(self) -> coo_array:
        """Given the generating monomials, infer conversion to Collins-Gisin 
        notation."""
        eq_row, eq_col, eq_data = [], [], []
        nof_equalities = 0
        if np.any(self._boolvec_for_FR_eqs):
            alternatives_as_boolarrays = {v: np.pad(r[:-1], ((1, 0), (0, 0)))
                                          for v, r in zip(
                    np.flatnonzero(self._boolvec_for_FR_eqs).flat,
                    self._CG_nonadjusting_ortho_groups_as_boolarrays)}
            alternatives_as_signs = {
                i: np.count_nonzero(bool_array, axis=1).astype(bool)
                for i, bool_array in alternatives_as_boolarrays.items()}

            for bool_vec in tqdm(self._monomials_as_lexboolvecs,
                    disable=not self.verbose,
                    desc="Discovering equalities   "):
                critical_boolvec_intersection = np.bitwise_and(bool_vec, self._boolvec_for_FR_eqs)
                if np.any(critical_boolvec_intersection):

                    absent_c_boolvec = bool_vec.copy()
                    absent_c_boolvec[critical_boolvec_intersection] = False
                    critical_values_in_boovec = np.flatnonzero(critical_boolvec_intersection)
                    signs = reduce(nb_outer_bitwise_xor,
                                   (alternatives_as_signs[i] for i in critical_values_in_boovec.flat))
                    adjustments = reduce(nb_outer_bitwise_or,
                                   (alternatives_as_boolarrays[i] for i in critical_values_in_boovec.flat))
                    terms_as_boolvecs = np.bitwise_or(
                        absent_c_boolvec[np.newaxis],
                        adjustments)
                    # Conversion from inequality to equality:
                    signs = np.hstack((signs,1))
                    terms_as_boolvecs = np.vstack((terms_as_boolvecs, 
                                                   bool_vec))
                    terms_as_rawidx = [self._raw_lookup_dict[boolvec.tobytes()] 
                                       for boolvec in terms_as_boolvecs]
                    terms_as_idxs = self.inverse[terms_as_rawidx]
                    true_signs = np.power(-1, signs)

                    eq_row.extend([nof_equalities] * len(signs))
                    eq_col.extend(terms_as_idxs.flat)
                    eq_data.extend(true_signs.flat)
                    nof_equalities += 1
            if self.verbose > 0:
                eprint("Number of nontrivial equality constraints in the LP:",
                        nof_equalities)
        return coo_array((eq_data, (eq_row, eq_col)),
                          shape=(nof_equalities, self.n_columns))

    @property
    def sparse_extra_equalities(self) -> coo_array:
        """Extra equalities in sparse matrix form."""
        eq_row, eq_col, eq_data = [], [], []
        nof_equalities = len(self.extra_equalities)
        for row_idx, eq in enumerate(self.extra_equalities):
            nof_vars = len(eq)
            eq_row.extend(np.repeat(row_idx, nof_vars))
            eq_col.extend([self.compmonomial_to_idx[x] for x in eq])
            eq_data.extend(eq.values())
        return coo_array((eq_data, (eq_row, eq_col)),
                          shape=(nof_equalities, self.n_columns))

    @property
    def sparse_equalities(self) -> coo_array:
        """All equalities (minimal and extra) in sparse matrix 
        form."""
        return vstack((self.minimal_sparse_equalities,
                       self.sparse_extra_equalities))

    @cached_property
    def minimal_sparse_inequalities(self) -> coo_array:
        """Here we express the nonnegativity of all `global` (maximal number
         of variables) events, converting the expressions into Collins-Gisin
         notation as needed."""
        ineq_row, ineq_col, ineq_data = [], [], []
        nof_inequalities = 0
        alternatives_as_boolarrays = {v: np.pad(r[1:], ((1, 0), (0, 0)))
                                      for v, r in zip(
                np.flatnonzero(self._boolvec_for_CG_ineqs).flat,
                self._CG_adjusting_ortho_groups_as_boolarrays)}
        alternatives_as_signs = {
            i: np.count_nonzero(bool_array, axis=1).astype(bool)
            for i, bool_array in alternatives_as_boolarrays.items()}
        for bool_vec in tqdm(self._monomials_as_lexboolvecs_non_CG,
                             disable=not self.verbose,
                             desc="Discovering inequalities   "):
            critical_boolvec_intersection = np.bitwise_and(bool_vec,
                                                           self._boolvec_for_CG_ineqs)
            if critical_boolvec_intersection.any():
                absent_c_boolvec = bool_vec.copy()
                absent_c_boolvec[critical_boolvec_intersection] = False
                critical_values_in_boovec = np.flatnonzero(
                    critical_boolvec_intersection)
                signs = reduce(nb_outer_bitwise_xor,
                               (alternatives_as_signs[i] for i in
                                critical_values_in_boovec.flat))
                adjustments = reduce(nb_outer_bitwise_or,
                                     (alternatives_as_boolarrays[i] for i in
                                      critical_values_in_boovec.flat))
                terms_as_boolvecs = np.bitwise_or(
                    absent_c_boolvec[np.newaxis],
                    adjustments)
                terms_as_rawidx = [self._raw_lookup_dict[term_boolvec.tobytes()]
                                   for term_boolvec in terms_as_boolvecs]
                terms_as_idxs = self.inverse[terms_as_rawidx]
                true_signs = np.power(-1, signs)

                ineq_row.extend([nof_inequalities] * len(signs))
                ineq_col.extend(terms_as_idxs.flat)
                ineq_data.extend(true_signs.flat)
            else:
                ineq_row.append(nof_inequalities)
                ineq_col.append(self.inverse[self._raw_lookup_dict[bool_vec.tobytes()]])
                ineq_data.append(1)
            nof_inequalities += 1
        return coo_array((ineq_data, (ineq_row, ineq_col)),
                          shape=(nof_inequalities, self.n_columns))

    @property
    def sparse_extra_inequalities(self) -> coo_array:
        """Extra inequalities in sparse matrix form."""
        ineq_row, ineq_col, ineq_data = [], [], []
        nof_inequalities = len(self.extra_inequalities)
        for row_idx, ineq in enumerate(self.extra_inequalities):
            nof_vars = len(ineq)
            ineq_row.extend(np.repeat(row_idx, nof_vars))
            ineq_col.extend([self.compmonomial_to_idx[x] for x in ineq])
            ineq_data.extend(ineq.values())
        return coo_array((ineq_data, (ineq_row, ineq_col)),
                          shape=(nof_inequalities, self.n_columns))

    @property
    def sparse_inequalities(self) -> coo_array:
        """All inequalities (minimal and extra) in sparse matrix form."""
        return vstack((self.minimal_sparse_inequalities,
                       self.sparse_extra_inequalities))

    def _coo_vec_to_mon_dict(self,
                             col: np.ndarray,
                             data: np.ndarray) -> Dict:
        """Convert a COO vector to a dictionary of monomials.
        
        Parameters
        ----------
        col : numpy.ndarray
            Column indices of the COO vector.
        data : numpy.ndarray
            Data of the COO vector.
            
        Returns
        -------
        Dict
            Dictionary of monomials.
        """
        return dict(zip(self.monomials[col].flat, data))

    def _coo_vec_to_name_dict(self, 
                              col: np.ndarray,
                              data: np.ndarray) -> Dict:
        """Convert a COO vector to a dictionary of monomial names.
        
        Parameters
        ----------
        col : numpy.ndarray
            Column indices of the COO vector.
        data : numpy.ndarray
            Data of the COO vector.
        
        Returns
        -------
        Dict
            Dictionary of monomial names.
        """
        return dict(zip(self.monomial_names[col].flat, data))

    def _coo_mat_to_dict(self,
                         input_coo_mat: coo_array,
                         string_keys: bool = False) -> List[Dict]:
        """Convert a COO matrix to a list of dictionaries.
        
        Parameters
        ----------
        input_coo_mat : coo_array
            Input COO matrix.
        string_keys : bool, optional
            Whether to use string keys or not, by default, ``False``.
        
        Returns
        -------
        List[Dict]
            List of dictionaries.
        """
        input_lil_mat = input_coo_mat.tolil(copy=False)
        args_iter = zip(input_lil_mat.rows, input_lil_mat.data)
        if string_keys:
            return [self._coo_vec_to_name_dict(*args) for args in args_iter]
        else:
            return [self._coo_vec_to_mon_dict(*args) for args in args_iter]

    def _mon_dict_to_coo_vec(self, monomials_dict: Dict) -> coo_array:
        """
        This is a PLACEHOLDER function, possibly to be deprecated, to convert
         dicts into COO matrices.
        """
        data = list(monomials_dict.values())
        keys = list(monomials_dict.keys())
        col = partsextractor(self.compmonomial_to_idx, keys)
        row = np.zeros(len(col), dtype=int)
        return coo_array((data, (row, col)), shape=(1, self.n_columns))

    @property
    def moment_equalities(self):
        """List of dictionaries of moment equalities."""
        return self._coo_mat_to_dict(self.sparse_equalities)

    @property
    def moment_inequalities(self):
        """List of dictionaries of moment inequalities."""
        return self._coo_mat_to_dict(self.sparse_inequalities)


    def _discover_inflation_orbits(self, 
                                   _raw_monomials_as_lexboolvecs: np.ndarray,
                                   raw_hash_table=False
                                   ) -> np.ndarray:
        """Calculates all the symmetries pertaining to the set of generating
        monomials. The new set of operators is a permutation of the old. The
        function outputs a list of all permutations.

        Returns
        -------
        numpy.ndarray[int]
            The orbits of the generating columns implied by the inflation
            symmetries.
        """
        nof_bitvecs_to_parse = len(_raw_monomials_as_lexboolvecs)
        if len(self.lexorder_symmetries) > 1:
            if not raw_hash_table:
                hash_table = {bitvec.tobytes(): i for i, bitvec in
                              enumerate(_raw_monomials_as_lexboolvecs)}
            else:
                hash_table = raw_hash_table
            non_identity_symmetries = self.lexorder_symmetries[1:]
            orbits = np.full(nof_bitvecs_to_parse, -1, dtype=int)
            for i in tqdm(range(nof_bitvecs_to_parse),
                          disable=not self.verbose,
                          total=nof_bitvecs_to_parse,
                          desc="Calculating orbits...             "):
                if orbits[i] == -1:
                    orbits[i] = i
                    initial = _raw_monomials_as_lexboolvecs[i]
                    initial_hash = initial.tobytes()
                    variants = initial[non_identity_symmetries]
                    variant_hashes = {variant.tobytes() for variant in variants}.difference({initial_hash})
                    remaining_orbit = []
                    for variant_hash in variant_hashes:
                        try:
                            remaining_orbit.append(hash_table[variant_hash])
                        except KeyError:
                            continue
                    orbits[remaining_orbit] = i
            return orbits
        else:
            return np.arange(nof_bitvecs_to_parse, dtype=int)

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
            self.semiknown_moments = dict()
        num_nontrivial_known = len(self.known_moments)
        if self.verbose > 1 and num_nontrivial_known > 1:
            eprint("Number of variables with fixed numeric value:",
                  num_nontrivial_known)
        if len(self.semiknown_moments):
            for k in self.known_moments.keys():
                self.semiknown_moments.pop(k, None)
        num_semiknown = len(self.semiknown_moments)
        if self.verbose > 1 and num_semiknown > 0:
            eprint(f"Number of semiknown variables: {num_semiknown}")

    def _reset_bounds(self) -> None:
        """Reset the lists of bounds."""
        self._reset_lowerbounds()
        self._reset_upperbounds()
        collect()

    def _reset_lowerbounds(self) -> None:
        """Reset the list of lower bounds."""
        self._reset_solution()
        self.moment_lowerbounds = dict()

    def _reset_upperbounds(self) -> None:
        """Reset the list of upper bounds."""
        self._reset_solution()
        self.moment_upperbounds = dict()

    def _reset_objective(self) -> None:
        """Reset the objective function."""
        self._reset_solution()
        self.objective = defaultdict(int)
        self.maximize = True  # Direction of the optimization

    def _reset_values(self) -> None:
        """Reset the known values."""
        self._reset_solution()
        self.known_moments     = dict()
        self.semiknown_moments = dict()
        self.known_moments[self.One] = 1.
        self.extra_equalities = []
        self.extra_inequalities = []
        collect()

    def _reset_solution(self) -> None:
        """Resets class attributes storing the solution to the LP
        relaxation."""
        for attribute in {"primal_objective",
                          "objective_value",
                          "solution_object"}:
            try:
                delattr(self, attribute)
            except AttributeError:
                pass
        self.status = "Not yet solved"


    ###########################################################################
    # Preparation for passing to external interfaces                          #
    ###########################################################################

    @cached_property
    def moment_equalities_by_name(self):
        """List of dictionaries of moment equalities by name."""
        return self._coo_mat_to_dict(self.sparse_equalities, string_keys=True)

    @cached_property
    def moment_inequalities_by_name(self):
        """List of dictionaries of moment inequalities by name."""
        return self._coo_mat_to_dict(self.sparse_inequalities, 
                                     string_keys=True)

    @cached_property
    def factorization_conditions_by_name(self):
        """Dictionary of factorization conditions by name."""
        return {mon.name: tuple(fac.name for fac in val)
                for mon, val in self.factorization_conditions.items()}

    @cached_property
    def sparse_factorization_conditions(self):
        """Dictionary of factorization conditions in sparse matrix form."""
        return {self.compmonomial_to_idx[mon]: partsextractor(self.compmonomial_to_idx, val)
                for mon, val in self.factorization_conditions.items()}

    @cached_property
    def quadratic_factorization_conditions_by_name(self):
        """Dictionary of quadratic factorization conditions by name."""
        return {mon.name: tuple(fac.name for fac in val) 
                for mon, val in self.quadratic_factorization_conditions.items()}

    @cached_property
    def sparse_quadratic_factorization_conditions(self):
        """Dictionary of quadratic factorization conditions in sparse matrix
        form."""
        return {self.compmonomial_to_idx[mon]: partsextractor(self.compmonomial_to_idx, val) 
                for mon, val in self.quadratic_factorization_conditions.items()}

    @property
    def sparse_objective(self) -> coo_array:
        """Sparse matrix representation of the objective function."""
        # TO BE DEPRECATED
        return self._mon_dict_to_coo_vec(self.objective)
    
    @property
    def objective_by_name(self) -> Dict:
        """Dictionary representation of the objective function."""
        return self._coo_mat_to_dict(self.sparse_objective,
                                     string_keys=True)[0]
    
    @property
    def sparse_known_vars(self) -> coo_array:
        """Sparse matrix representation of the known values."""
        # TO BE DEPRECATED
        return self._mon_dict_to_coo_vec(self.known_moments)
    
    @property
    def known_vars_by_name(self) -> Dict:
        """Dictionary representation of the known values."""
        return self._coo_mat_to_dict(self.sparse_known_vars, 
                                     string_keys=True)[0]
        
    @property
    def sparse_lowerbounds(self) -> coo_array:
        """Sparse matrix representation of the lower bounds."""
        # TO BE DEPRECATED
        return self._mon_dict_to_coo_vec(self.moment_lowerbounds)
    
    @property
    def lowerbounds_by_name(self) -> Dict:
        """Dictionary representation of the lower bounds."""
        return self._coo_mat_to_dict(self.sparse_lowerbounds, 
                                     string_keys=True)[0]
        
    @property
    def sparse_upperbounds(self) -> coo_array:
        """Sparse matrix representation of the upper bounds."""
        # TO BE DEPRECATED
        return self._mon_dict_to_coo_vec(self.moment_upperbounds)
    
    @property
    def upperbounds_by_name(self) -> Dict:
        """Dictionary representation of the upper bounds."""
        return self._coo_mat_to_dict(self.sparse_upperbounds, 
                                     string_keys=True)[0]

    @property
    def sparse_semiknown(self) -> coo_array:
        """Sparse matrix representation of the semiknown values."""
        nof_semiknown = len(self.semiknown_moments)
        nof_variables = len(self.compmonomial_to_idx)
        row = np.repeat(np.arange(nof_semiknown), 2)
        col = [(self.compmonomial_to_idx[x], self.compmonomial_to_idx[x2])
               for x, (c, x2) in self.semiknown_moments.items()]
        col = list(sum(col, ()))
        data = [(1, -c) for x, (c, x2) in self.semiknown_moments.items()]
        data = list(sum(data, ()))
        return coo_array((data, (row, col)),
                          shape=(nof_semiknown, nof_variables))

    @property
    def semiknown_by_name(self) -> Dict:
        """Dictionary representation of the semiknown values."""
        return {mon.name: (coeff, subs.name)
                                         for mon, (coeff, subs)
                                         in self.semiknown_moments.items()}

    def _prepare_solver_matrices(self,
                                 separate_bounds: bool = True) -> dict:
        """Convert arguments from dictionaries to sparse coo_array form to
        pass to the solver.

        Parameters
        ----------
        separate_bounds : bool, optional
            Whether to have variable bounds as a separate item in
            ``solverargs`` (True) or absorb them into the inequalities (False).
            By default, ``True``.

        Returns
        -------
        dict
            The arguments to be passed to the solver.

        Raises
        ------
        Exception
            If the LP has not been generated yet.
        """
        if not self._lp_has_been_generated:
            raise Exception("LP is not generated yet. " +
                            "Call \"InflationLP._generate_lp()\" first")

        assert set(self.known_moments.keys()).issubset(self.monomials), \
            ("Error: Tried to assign known values outside of the variables: " +
             str(set(self.known_moments.keys()
                     ).difference(self.monomials)))

        internal_equalities = vstack((self.sparse_equalities,
                                      self.sparse_semiknown))

        solverargs = {"objective": self.sparse_objective,
                      "known_vars": self.sparse_known_vars,
                      "equalities": internal_equalities,
                      "inequalities": self.sparse_inequalities,
                      "variables": self.monomial_names}
        if separate_bounds:
            solverargs["lower_bounds"] = self.sparse_lowerbounds
            solverargs["upper_bounds"] = self.sparse_upperbounds
        else:
            lb_mat = expand_sparse_vec(self.sparse_lowerbounds,
                                       conversion_style="lb")
            ub_mat = expand_sparse_vec(self.sparse_upperbounds,
                                       conversion_style="ub")
            solverargs["inequalities"] = vstack(
                (self.sparse_inequalities, lb_mat, ub_mat))
        return solverargs

    def _prepare_solver_arguments(self, 
                                  separate_bounds: bool = True) -> dict:
        """Prepare arguments to pass to the solver.

        The solver takes as input the following arguments, which are all
        dicts with keys as scalar LP variables:
            * "objective": dict with values the coefficient of the key
            variable in the objective function.
            * "known_vars": scalar variables that are fixed to be constant.
            * "semiknown_vars": if applicable, linear proportionality
            constraints between variables in the LP.
            * "equalities": list of dicts where each dict gives the
            coefficients of the keys in a linear equality constraint.
            * "inequalities": list of dicts where each dict gives the
            coefficients of the keys in a linear inequality constraint.

        Parameters
        ----------
        separate_bounds : bool, optional
            Whether to have variable bounds as a separate item in
            ``solverargs`` (True) or absorb them into the inequalities (False).
            By default, ``True``.

        Returns
        -------
        dict
            A tuple with the arguments to be passed to the solver.

        Raises
        ------
        Exception
            If the LP relaxation has not been calculated yet.
        """
        if not self._lp_has_been_generated:
            raise Exception("LP is not generated yet. " +
                            "Call \"InflationLP._generate_lp()\" first")

        assert set(self.known_moments.keys()).issubset(self.monomials), \
            ("Error: Tried to assign known values outside of the variables: " +
             str(set(self.known_moments.keys()
                     ).difference(self.monomials)))

        # Rectification in the event of unnormalized problem
        variables = self.monomial_names.tolist()
        if {1, self.constant_term_name}.isdisjoint(self.known_vars_by_name):
            self.known_vars_by_name[self.constant_term_name] = 1.
            if self.constant_term_name not in variables:
                variables.append(self.constant_term_name)
        solverargs = {"objective": self.objective_by_name,
                      "known_vars": self.known_vars_by_name,
                      "semiknown_vars": self.semiknown_by_name,
                      "equalities": self.moment_equalities_by_name,
                      "inequalities": self.moment_inequalities_by_name,
                      "variables": variables}
        if separate_bounds:
            solverargs["lower_bounds"] = self.lowerbounds_by_name
            solverargs["upper_bounds"] = self.upperbounds_by_name
        else:
            solverargs["inequalities"].extend({mon.name: 1, '1': -bnd}
                                              for mon, bnd in
                                              self.moment_lowerbounds.items())
            solverargs["inequalities"].extend({mon.name: -1, '1': bnd}
                                              for mon, bnd in
                                              self.moment_upperbounds.items())
        return solverargs

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

    def _set_upperbounds(self, upperbounds: Union[dict, None]) -> None:
        """Set upper bounds for variables in the LP relaxation.

        Parameters
        ----------
        upperbounds : Union[dict, None]
            Dictionary with keys as moments and values as upper bounds. The
            keys can be either strings, instances of `CompoundMonomial` or
            moments encoded as 2D arrays.
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
        self.moment_upperbounds = sanitized_upperbounds

    def _set_lowerbounds(self, lowerbounds: Union[dict, None]) -> None:
        """Set lower bounds for variables in the LP relaxation.

        Parameters
        ----------
        lowerbounds : Union[dict, None]
            Dictionary with keys as moments and values as upper bounds. The
            keys can be either strings, instances of `CompoundMonomial` or
            moments encoded as 2D arrays.
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
        self.moment_lowerbounds = sanitized_lowerbounds

    def mon_to_boolvec(self, mon: np.ndarray) -> np.ndarray:
        """Convert a monomial to a boolean vector.
        
        Parameters
        ----------
        mon : numpy.ndarray
            Monomial as a 2D array.
        
        Returns
        -------
        numpy.ndarray
            Boolean vector.
        """
        boolvec = self.blank_bool_vec.copy()
        boolvec[self.mon_to_lexrepr(mon)] = True
        return boolvec
