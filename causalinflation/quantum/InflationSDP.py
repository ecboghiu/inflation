# import copy
import itertools
import numpy as np
import pickle
import sympy as sp
from collections import Counter, defaultdict

from causalinflation import InflationProblem
from causalinflation.quantum.general_tools import (to_name, to_representative,
                                            to_numbers, mul,
                                            transform_vars_to_symb,
                                            substitute_variable_values_in_monlist,
                                            substitute_sym_with_numbers,
                                            string2prob,
                                            phys_mon_1_party_of_given_len,
                                            is_knowable, is_physical,
                                            # label_knowable_and_unknowable,
                                            monomialset_name2num,
                                            monomialset_num2name,
                                            factorize_monomials,
                                            factorize_monomial,
                                            find_permutation,
                                            apply_source_permutation_coord_input,
                                            from_numbers_to_flat_tuples,
                                            generate_commuting_measurements,
                                            generate_noncommuting_measurements,
                                            from_coord_to_sym,
                                            clean_coefficients,
                                            dimino_sympy)
from causalinflation.quantum.fast_npa import calculate_momentmatrix
from causalinflation.quantum.monomial_class import Monomial
from causalinflation.quantum.sdp_utils import solveSDP_MosekFUSION2
from causalinflation.quantum.writer_utils import (write_to_csv, write_to_mat,
                                                  write_to_sdpa)
# ncpol2sdpa >= 1.12.3 is required for quantum problems to work
from ncpol2sdpa import flatten, projective_measurement_constraints
from ncpol2sdpa.nc_utils import apply_substitutions, simplify_polynomial
from typing import List, Dict, Union, Tuple, Any
from warnings import warn



try:
    from tqdm import tqdm
except ImportError:
    def tqdm(*args, **kwargs):
        return args[0]

class InflationSDP(object):
    """Class for generating and solving an SDP relaxation for quantum inflation.

    Parameters
    ----------
    InflationProblem : InflationProblem
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
    def __init__(self, InflationProblem: InflationProblem,
                       commuting: bool = False,
                       verbose: int = 0):
        """Constructor for the InflationSDP class.
        """

        self.verbose = verbose
        self.commuting = commuting
        self.InflationProblem = InflationProblem
        self.names = self.InflationProblem.names
        if self.verbose > 1:
            print(self.InflationProblem)
        self.measurements, self.substitutions, self.names \
                                            = self._generate_parties()

        self.nr_parties = len(self.names)
        self.hypergraph = self.InflationProblem.hypergraph
        self.inflation_levels = self.InflationProblem.inflation_level_per_source
        self.outcome_cardinalities = self.InflationProblem.outcomes_per_party
        self.setting_cardinalities = self.InflationProblem.settings_per_party
        self.maximize = True    # Direction of the optimization
        self.split_node_model = self.InflationProblem.split_node_model
        self.is_knowable_q_split_node_check = self.InflationProblem.is_knowable_q_split_node_check

    def atomic_knowable_q(self, atomic_monomial):
        first_test = is_knowable(atomic_monomial)
        if first_test and self.split_node_model:
            # minimal_monomial = tuple(tuple(vec) for vec in np.take(atomic_monomial, [0, -1, 2], axis=1))
            minimal_monomial = tuple(tuple(vec) for vec in np.take(atomic_monomial, [0, -2, -1], axis=1))
            return self.is_knowable_q_split_node_check(minimal_monomial)
        else:
            return first_test

    def inflation_aware_to_representative(self, mon: np.ndarray):
        return tuple(tuple(vec) for vec in to_representative(mon,
                                 inflevels=self.inflation_levels,
                                 commuting=self.commuting))

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
        self.use_lpi_constraints = False

        # Process the column_specification input and store the result
        # in self.generating_monomials.
        self.generating_monomials_sym, self.generating_monomials = \
                        self.build_columns(column_specification,
                            return_columns_numerical=True)

        if self.verbose > 0:
            print("Number of columns:", len(self.generating_monomials))

        # Calculate the moment matrix without the inflation symmetries.
        self.unsymmetrized_mm_idxs, self.unsymidx_to_canonical_mon_dict = self._build_momentmatrix()
        if self.verbose > 1:
            print("Number of variables before symmetrization:",
                  len(self.unsymidx_to_canonical_mon_dict))

        # Calculate the inflation symmetries.
        self.inflation_symmetries = self._calculate_inflation_symmetries()

        # Apply the inflation symmetries to the moment matrix.
        # TODO: make it so that after this 'monomials_list' is no longer
        # needed and can be safely deleted. Currently this list can get big
        # which just occupies needless memory. Also currently its only used
        # for the objective function, because the user might input some
        # monomial that is not in the form of the ones found in
        # 'remaining_monomials'. The correct thing to do is to use a function
        # to bring the monomials in the user-inputted objective function
        # to a canonoical form! But this is not implemented yet.
        self.momentmatrix, self.orbits, self.symidx_to_canonical_mon_dict \
                    = self._apply_inflation_symmetries(self.unsymmetrized_mm_idxs,
                                                       self.unsymidx_to_canonical_mon_dict,
                                                       self.inflation_symmetries)

        self.largest_moment_index = max(self.symidx_to_canonical_mon_dict.keys())

        #NOTE: self.symidx_to_canonical_mon_dict is meant to replace 'remaining monomials'
        #NOTE: self.symmetrized_mm_idxs is meant to replace


        # # Associate the names of all copies to the same variable
        # for key, val in self._mon_string2int.items():
        #     self._mon_string2int[key] = orbits[val]
        #
        # # Bring remaining monomials to a representative form, and add the
        # # corresponding identifications to the dictionary
        # for idx, [var, mon] in enumerate(tqdm(
        #                                monomialset_name2num(remaining_monomials,
        #                                                     self.names),
        #                                       disable=not self.verbose,
        #                                   desc="Computing canonical forms   ")):
        #     canonical = to_name(to_representative(np.array(mon),
        #                                           self.inflation_levels,
        #                                           self.commuting),
        #                         self.names)
        #     remaining_monomials[idx][1] = canonical
        #     self._mon_string2int[canonical] = var


        self.symidx_to_Monomials_dict = {k: Monomial(v,
                                                     atomic_is_knowable=self.atomic_knowable_q,
                                                     sandwich_positivity=True) for (k, v) in self.symidx_to_canonical_mon_dict.items()}
        ### HOW DOES to_representative work without knowing inflation level per source?? This seems like a really bad idea.

        if self.commuting:
            self.physical_monomials = set(range(len(self.symidx_to_Monomials_dict))) #TODO: rename to ...monomial_idxs
        else:
            self.physical_monomials = set([k for (k, Mon) in self.symidx_to_Monomials_dict.items() if Mon.physical_q])

        self.knowable_moments = {k: list(map(self.InflationProblem.rectify_fake_setting_atomic_factor, Mon.knowable_factors))
                                 for (k, Mon) in self.symidx_to_Monomials_dict.items() if Mon.knowability_status == 'Yes'}

        knowability_statusus = np.empty((self.largest_moment_index + 1,), dtype='<U4')
        knowability_statusus[[0, 1]] = 'Yes'
        for i, monomial in self.symidx_to_Monomials_dict.items():
            knowability_statusus[i] = monomial.knowability_status
        # self.knowability_statusus = [monomial.knowability_status for monomial in self.symidx_to_Monomials_dict.values()]
        _counter = Counter(knowability_statusus)
        self._n_known = _counter['Yes'] - 2 #TODO: rename all these 3 as knowable instead of known.
        self._n_something_known = _counter['Semi']
        self._n_unknown = _counter['No']
        # self.reordering_of_monomials = np.hstack((
        #         np.flatnonzero(self.knowability_statusus == 'Yes'),
        #         np.flatnonzero(self.knowability_statusus == 'Semi'),
        #         np.flatnonzero(self.knowability_statusus == 'No')))

        self._all_atomic_knowable = set(itertools.chain.from_iterable(mon.knowable_factors for mon in self.symidx_to_Monomials_dict.values()))




        #TODO: Restore self._monomials_list_all, as used in 'set_objective' (probably best to fix 'set_objective')
        #TODO: Restore 'monomials_list' and 'semiknown_reps' for use in 'set_distribution'

        # Define trivial arrays for distribution and objective

        #TODO: self.known_moments can have two forms: dict, and sparse array.
        #TODO: self.semiknown_moments can have two forms: dict {x: (c, x2), ...}, and sparse array.
        #TODO: self.objective_as_dict can have two forms: dict {x: v, ...}, and sparse array.
        self.known_moments      = {0: 0., 1: 1.}
        self.semiknown_moments  = dict()
        self.objective          = 0.
        self._objective_as_dict = {1: 0.}

        # For the bounds, monomials should be hashed in the same way as
        # self.known_moments, self._objective_as_dict, etc.
        self.moment_linear_equalities = []
        self.moment_linear_inequalities = []
        self.moment_lowerbounds = {physical: 0 for physical in self.physical_monomials}
        self.moment_upperbounds = {}



    def set_distribution(self,
                         prob_array: np.ndarray,
                         use_lpi_constraints: bool = False) -> None:
        """Set numerically the knowable moments and semiknowable moments according
        to the probability distribution specified. If p is None, or the user
        doesn't pass any argument to set_distribution, then this is understood
        as a request to delete information about past distributions. If prob_array contains
        elements that are either None or nan, then this is understood as leaving
        the corresponding variable free in the SDP approximation.
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
        _pdims = len(list(prob_array.shape))
        assert _pdims % 2 == 0, "The probability distribution must have equal number of inputs and outputs"
        # list(p.shape[:int(_pdims/2)]
        #      ) == self.InflationProblem.outcomes_per_party
        # list(p.shape[int(_pdims/2):]
        #      ) == self.InflationProblem.settings_per_party

        self.use_lpi_constraints = use_lpi_constraints

        if (len(self._objective_as_dict) > 1) and self.use_lpi_constraints:
            warn("You have an objective function set. Be aware that imposing " +
                 "linearized polynomial constraints will constrain the " +
                 "optimization to distributions with fixed marginals.")

        # atomic_knowable_variables = [entry
        #         for idx, entry in enumerate(self.monomials_list[:self._n_known])
        #                            if len(self.semiknowable_atoms[idx][1]) == 1]
        # atomic_numerical_values = {var: compute_numeric_value(var[1],
        #                                                          prob_array,
        #                                             self.InflationProblem.names)
        #                            for var in self._all_atomic_knowable}
        knowable_atoms_dict = defaultdict(lambda x: np.nan)
        for atom in self._all_atomic_knowable:
            knowable_atoms_dict[atom] = compute_marginal(prob_array, atom)

        for mon in self.symidx_to_Monomials_dict.values():
            if mon.knowability_status == 'No':
                mon.known_part = 1
                mon.unknown_part = self.inflation_aware_to_representative(np.vstack(mon.unknowable_factors)) #TODO: Use global function
            else:
                valuation_of_knowable_part = [knowable_atoms_dict[knowable_atom] for knowable_atom in mon.knowable_factors]
                if 0. in valuation_of_knowable_part:
                    mon.known_part = 0
                    mon.unknown_part = tuple()
                    mon.knowability_status = 'Yes'
                else:
                    actually_known_factors = np.logical_not(np.isnan(valuation_of_knowable_part))
                    nof_known_factors = np.count_nonzero(actually_known_factors)
                    if nof_known_factors == mon.nof_factors:
                        mon.known_part = np.prod(valuation_of_knowable_part)
                        mon.unknown_part = tuple()
                        mon.knowability_status = 'Yes'
                    else:
                        if nof_known_factors == 0:
                            mon.knowability_status = 'No'
                        else:
                            mon.knowability_status = 'Semi'
                        mon.known_part = np.prod(np.compress(
                            actually_known_factors,
                            valuation_of_knowable_part))
                        knowable_factors_which_are_not_known = [factor for (factor, known_status) in zip(
                                mon.knowable_factors_uncompressed,
                                actually_known_factors) if not known_status]
                        mon.unknown_part = self.inflation_aware_to_representative(np.vstack((
                            mon.factors_as_block(knowable_factors_which_are_not_known),
                            mon.factors_as_block(mon.unknowable_factors),
                        )))
        #TODO: Finish the function. Collect all unique unknown parts, form relationship matrices.


    def set_objective(self,
                      objective: sp.core.symbol.Symbol,
                      direction: str ='max') -> None:
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

        if self.use_lpi_constraints:
            warn("You have the flag `use_lpi_constraints` set to True. Be " +
                 "aware that imposing linearized polynomial constraints will " +
                 "constrain the optimization to distributions with fixed " +
                 "marginals.")

        self.objective = objective

        if (sp.S.One*objective).free_symbols:
            # Sanitize input: pass through substitutions to get
            # objects that we have used previously
            objective = simplify_polynomial(sp.expand(objective),
                                            self.substitutions)
            # # Build string-to-variable dictionary
            # string2int_dict = {**{'0': '0', '1': '1'},
            #                    **dict(self._monomials_list_all[:, ::-1])}

            #Build monomial to index dictionary
            self.symidx_from_Monomials_dict = {self.inflation_aware_to_representative(v.as_ndarray): k for (k, v) in self.symidx_to_Monomials_dict.items()}
            acceptable_monomials = set(self.symidx_from_Monomials_dict.keys())

            # Express objective in terms of representatives
            symmetrized_objective = {1: 0.}
            for monomial, coeff in objective.as_coefficients_dict().items():
                monomial_as_str = str(monomial)
                if monomial_as_str == '1':
                    symmetrized_objective[1] += sign*coeff
                else:
                    monomial_as_array = to_numbers(monomial_as_str, self.names)
                    monomial_as_repr_array = self.inflation_aware_to_representative(np.asarray(monomial_as_array))
                    assert monomial_as_repr_array in acceptable_monomials, 'Monomial specified does not appear in our moment matrix.'
                    repr =  self.symidx_from_Monomials_dict[monomial_as_repr_array]
                    # If the objective contains a known value add it to the constant
                    if repr in self.known_moments.keys():
                        symmetrized_objective[1] += \
                                                sign*coeff*self.known_moments[repr]
                    elif repr in symmetrized_objective.keys():
                        symmetrized_objective[repr] += sign * coeff
                    else:
                        symmetrized_objective[repr] = sign * coeff
            self._objective_as_dict = symmetrized_objective
        else:
            self._objective_as_dict = {1: sign * float(objective)}

    def set_values(self,
                   values: Dict[Any, float],
                   use_lpi_constraints: bool = False,
                   only_specified_values: bool = False,
                   ) -> None:
        """Directly assign numerical values to variables in the moment matrix.
        This is done via a dictionary where keys are the variables to have
        numerical values assigned (either in their operator form, in string
        form, or directly referring to the variable in the moment matrix), and
        the values are the corresponding numerical quantities.

        Parameters
        ----------
        values : Dict[Union[simpy.core.symbol.Symbol, int, str], float]
            The description of the variables to be assigned numerical values and
            the corresponding values.

        use_lpi_constraints : bool
            Specification whether linearized polynomial constraints (see, e.g.,
            Eq. (D6) in arXiv:2203.16543) will be imposed or not.

        only_specified_values : bool
            Specification whether one wishes to fix only the variables provided,
            or also the variables containing products of the monomials fixed.
        """
        self.use_lpi_constraints = use_lpi_constraints
        self.known_moments = {0: 0., 1: 1.}
        names_to_vars = dict(self.monomials_list[:, ::-1])
        for key, val in values.items():
            if type(key) == int:
                self.known_moments[key] = val
            elif any([type(key) == sp.core.symbol.Symbol,
                      type(key) == sp.physics.quantum.operator.HermitianOperator,
                      type(key) == str]):
                try:
                    key_int = names_to_vars[str(key)]
                    self.known_moments[key_int] = val
                except KeyError:
                    raise Exception(f"The monomial {key} could not be found in "
                                    + "the moment matrix. Please input it in "
                                    + "the form as it appears in "
                                    + "self.monomials_list.")
            else:
                raise Exception(f"The type of the monomial {key} to be "
                                + "assigned is not understood. Please use "
                                + "the integer associated to the monomial, "
                                + "its product of Sympy symbols, or the string "
                                + "representing the latter.")
        if not only_specified_values:
            # Assign numerical values to products of known atoms
            for var, monomial_factors in self.semiknowable_atoms:
                numeric_factors = np.array([values.get(factor, factor)
                                            for factor in monomial_factors],
                                           dtype=object)
                if all(numeric_factors <= 1.):
                    # When all atoms have numerical values the variable is known
                    self.known_moments[var] = np.prod(numeric_factors)
                elif self.use_lpi_constraints:
                    if np.sum(numeric_factors > 1) == 1:
                        # Compute the semiknown
                        sorted_factors = np.sort(numeric_factors)
                        self.semiknown_moments[var] = [sorted_factors[:-1].prod(),
                                                       sorted_factors[-1]]
                    elif ((np.sum(numeric_factors > 1) > 1)
                        and (np.sum(numeric_factors <= 1.) > 0)):
                        pos = np.where([mon[0] for mon in self.monomials_list]
                                       == var)[0]
                        monomial = self.monomials_list[pos][1]
                        warn(f"The variable {var}, corresponding to monomial "
                             + f"{monomial}, factorizes into a known part times"
                             + " a product of variables. Recombining these "
                             + "variables is not yet supported, so the variable"
                             + " will be treated as unknown.")
        if self.objective != 0:
            self._update_objective()    #PASS THE SEMIKNOWNS THROUGH THE OBJECTIVE?

    def solve(self, interpreter: str='MOSEKFusion',
                    feas_as_optim: bool=False,
                    dualise: bool=True,
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
        if feas_as_optim and len(self._objective_as_dict) > 1:
            warn("You have a non-trivial objective, but set to solve a " +
                 "feasibility problem as optimization. Setting "
                 + "feas_as_optim=False and optimizing the objective...")
            feas_as_optim = False

        # solveSDP_arguments = {"positionsmatrix":  self.momentmatrix,
        #                       "objective":        self._objective_as_dict,
        #                       "known_vars":       self.known_moments,
        #                       "semiknown_vars":   self.semiknown_moments,
        #                       "positive_vars":    self.physical_monomials,
        #                       "feas_as_optim":    feas_as_optim,
        #                       "verbose":          self.verbose,
        #                       "solverparameters": solverparameters}
        solveSDP_arguments = {"positionsmatrix":  self.momentmatrix,
                              "objective":        self._objective_as_dict,
                              "known_vars":       self.known_moments,
                              "semiknown_vars":   self.semiknown_moments,
                              "positive_vars":    self.physical_monomials,
                              "feas_as_optim":    feas_as_optim,
                              "verbose":          self.verbose,
                              "solverparameters": solverparameters,
                              "var_lowerbounds":  self.moment_lowerbounds,
                              "var_upperbounds":  self.moment_upperbounds,
                              "var_equalities":   self.moment_linear_equalities,
                              "var_inequalities": self.moment_linear_inequalities,
                              "solve_dual":       dualise}

        # self.solution_object, lambdaval, self.status = \
        #                               solveSDP_MosekFUSION(**solveSDP_arguments)
        self.solution_object, lambdaval, self.status = \
                                    solveSDP_MosekFUSION2(**solveSDP_arguments)

        # Process the solution
        if self.status == 'feasible':
            self.primal_objective = lambdaval
            self.objective_value  = lambdaval * (1 if self.maximize else -1)

        # Process the dual certificate in a generic form
        if self.status in ['feasible', 'infeasible']:
            coeffs      = self.solution_object['dual_certificate']
            names       = [self.monomials_list[idx-2,1] for idx in coeffs.keys()
                            if idx > 1]    # We'd probably want this cleaner
            clean_names = np.concatenate((['0', '1'], names))
            self.dual_certificate = np.array(list(zip([0]+list(coeffs.values()),
                                                      clean_names)),
                                             dtype=object)
            self.dual_certificate_lowerbound = 0

    def certificate_as_probs(self, clean: bool=False,
                             chop_tol: float=1e-10,
                             round_decimals: int=3) -> sp.core.symbol.Symbol:
        """Give certificate as symbolic sum of probabilities that is greater
        than or equal to 0.

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
            coeffs = self.dual_certificate[:, 0].astype(float)
            names  = self.dual_certificate[:, 1]
        except AttributeError:
            raise Exception("For extracting a certificate you need to solve " +
                            "a problem. Call 'InflationSDP.solve()' first")
        if len(self.semiknown_moments) > 0:
            warn("Beware that, because the problem contains linearized " +
                 "polynomial constraints, the certificate is not guaranteed " +
                 "to apply to other distributions")
        # C: why did I write this??
        # names can still contain duplicated names, so we need to remove them
        # new_dual_certificate = {tuple(name): 0 for name in names}
        # for i, name in enumerate(names):
        #     new_dual_certificate[tuple(name)] += coeffs[i]
        # coeffs = np.array(list(new_dual_certificate.values()))
        # names = list(new_dual_certificate.keys())

        if clean and not np.allclose(coeffs, 0):
            coeffs = clean_coefficients(coeffs, chop_tol, round_decimals)

        # Quickfix to a bug introduced in some committ, we need this for the
        # certificates
        from causalinflation.quantum.general_tools import factorize_monomial, to_numbers, to_name
        names_factorised = []
        for mon in names[2:]:
            factors = factorize_monomial(to_numbers(mon, self.names))
            factors_name = [to_name(factor, self.names) for factor in factors]
            names_factorised.append(factors_name)
        names = names[:2].tolist() + names_factorised

        polynomial = 0
        for i, row in enumerate(names):
            asprobs = [string2prob(
                term, self.InflationProblem.nr_parties) for term in row]
            monomial = sp.S.One
            for prob in asprobs:
                monomial *= prob
            monomial *= coeffs[i]
            polynomial += monomial
        return polynomial

    def certificate_as_objective(self, clean: bool=False,
                                 chop_tol: float=1e-10,
                                 round_decimals: int=3) -> sp.core.symbol.Symbol:
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
            coeffs = self.dual_certificate[:, 0].astype(float)
            names = self.dual_certificate[:, 1]
        except AttributeError:
            raise Exception("For extracting a certificate you need to solve " +
                            "a problem. Call 'InflationSDP.solve()' first")
        if len(self.semiknown_moments) > 0:
            warn("Beware that, because the problem contains linearized " +
                 "polynomial constraints, the certificate is not guaranteed " +
                 "to apply to other distributions")
        if clean and not np.allclose(coeffs, 0):
            coeffs = clean_coefficients(coeffs, chop_tol, round_decimals)

        # Quickfix to a bug introduced in some committ, we need this for the
        # certificates
        from causalinflation.quantum.general_tools import factorize_monomial, to_numbers, to_name
        names_factorised = []
        for mon in names[2:]:
            factors = factorize_monomial(to_numbers(mon, self.names))
            factors_name = [to_name(factor, self.names) for factor in factors]
            names_factorised.append(factors_name)
        names = names[:2].tolist() + names_factorised

        polynomial = 0
        for i, row in enumerate(names):
            monomial = sp.S.One
            for name in row:
                if name == '1':
                    monomial = sp.S.One
                elif name == '0':
                    monomial = sp.S.Zero
                else:
                    letters = name.split('*')
                    for letter in letters:
                        op = sp.Symbol(letter, commutative=False)
                        monomial *= op
            monomial *= coeffs[i]
            polynomial += monomial
        return polynomial

    def certificate_as_correlators(self,
                                   clean: bool=False,
                                   chop_tol: float=1e-10,
                                   round_decimals: int=3) -> sp.core.symbol.Symbol:
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
        if not all([o == 2 for o in self.InflationProblem.outcomes_per_party]):
            raise Exception("Correlator certificates are only available " +
                            "for 2-output problems")
        try:
            coeffs = self.dual_certificate[:, 0].astype(float)
            names  = self.dual_certificate[:, 1]
        except AttributeError:
            raise Exception("For extracting a certificate you need to solve " +
                            "a problem. Call 'InflationSDP.solve()' first")
        if len(self.semiknown_moments) > 0:
            warn("Beware that, because the problem contains linearized " +
                 "polynomial constraints, the certificate is not guaranteed " +
                 "to apply to other distributions")

        if clean and not np.allclose(coeffs, 0):
            coeffs = clean_coefficients(coeffs, chop_tol, round_decimals)

        # Quickfix to a bug introduced in some committ, we need this for the
        # certificates
        from causalinflation.quantum.general_tools import factorize_monomial, to_numbers, to_name
        names_factorised = []
        for mon in names[2:]:
            factors = factorize_monomial(to_numbers(mon, self.names))
            factors_name = [to_name(factor, self.names) for factor in factors]
            names_factorised.append(factors_name)
        names = names[:2].tolist() + names_factorised

        polynomial = 0
        for i, row in enumerate(names):
            poly1 = sp.S.One
            if row[0] != '1' and row[0] != '0':
                for name in row:
                    factors = name.split('*')
                    #correlator = '\langle '

                    aux_prod = sp.S.One
                    for factor_name in factors:
                        simbolo = sp.Symbol(factor_name[0]+'_{'+factor_name[-3]+'}', commuting=True)
                        projector = sp.Rational(1, 2)*(sp.S.One - simbolo)
                        aux_prod *= projector
                    aux_prod = sp.expand(aux_prod)
                    # Now take products and make them a single variable to make them 'sticking togetther' easier
                    suma = 0
                    for var, coeff in aux_prod.as_coefficients_dict().items():
                        if var == sp.S.One:
                            expected_value = sp.S.One
                        else:
                            if str(var)[-3:-1] == '**':
                                base, exp = var.as_base_exp()
                                auxname = '<' + ''.join(str(base).split('*')) + '>'
                                auxname = '\langle ' + ''.join(str(base).split('*')) + ' \\rangle'
                                base = sp.Symbol(auxname, commutative=True)
                                expected_value = base ** exp
                            else:
                                auxname = '<'+ ''.join(str(var).split('*')) + '>'
                                auxname = '\langle ' + ' '.join(str(var).split('*')) + ' \\rangle'
                                expected_value = sp.Symbol(auxname, commutative=True)
                        suma += coeff*expected_value
                    poly1 *= suma
                else:
                    if row[0] == '1':
                        poly1 = sp.S.One
                    elif row[0] == '0':
                        poly1 = sp.S.Zero
            polynomial += coeffs[i]*poly1

        return sp.expand(polynomial)


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
            label = filename[:-len(extension)-1]
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
                      return_columns_numerical: bool = False) -> None:
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
            if type(column_specification[0]) == list or type(column_specification[0]) == np.ndarray:
                if len(np.array(column_specification[1]).shape) == 2:
                    # If we are here, then the input to build columns is a list
                    # of monomials in array form, so we just return this
                    # e.g., [[0], [[1, 1, 1, 0, 0, 0]], [[1, 1, 1, 0, 0, 0],[2, 2, 1, 0, 0, 0]], ...]
                    # or [np.array([0]), np.array([[1, 1, 1, 0, 0, 0]]), np.array([[1, 1, 1, 0, 0, 0],[2, 2, 1, 0, 0, 0]]), ...]
                    columns = [np.array(mon, dtype=np.uint8) for mon in column_specification]
                elif len(np.array(column_specification[1]).shape) == 1:
                    # If the depth of column_specification is just 2,
                    # then the input must be in the form of
                    # e.g., [[], [0], [1], [0, 0]] -> {1, A{:}, B{:}, (A*A){:}}
                    # which just specifies the party structure in the
                    # generating set
                    columns = self._build_cols_from_col_specs(column_specification)
            else:
                columns = []
                for col in column_specification:
                    if col == sp.S.One or col == 1:
                        columns += [np.array([0],dtype=np.uint8)]
                    else:
                        columns += [np.array(to_numbers(str(col), self.names),dtype=np.uint8)]
        elif type(column_specification) == str:
            if 'npa' in column_specification.lower():
                npa_level = int(column_specification[3:])
                col_specs = [[]]
                max_length = max_monomial_length if max_monomial_length > 0 and max_monomial_length < npa_level else npa_level
                for length in range(1, max_length+1):
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
                for pfreq in itertools.product(*[range(local_level+1) for _ in range(self.nr_parties)]):
                    if sum(pfreq) <= max_length:
                        party_frequencies.append(list(reversed(pfreq)))
                party_frequencies = sorted(party_frequencies, key=sum)

                col_specs = []
                for pfreq in party_frequencies:
                    lst = []
                    for party in range(self.nr_parties):
                        lst += [party]*pfreq[party]
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
                        physmon_lens = [inf_level]*self.InflationProblem.nr_sources
                    else:
                        physmon_lens = [int(inf_level)
                                        for inf_level in column_specification[8:]]
                    max_total_mon_length = sum(physmon_lens)
                except:
                    # If no numbers come after, by default we use all physical operators
                    physmon_lens = self.InflationProblem.inflation_level_per_source
                    max_total_mon_length = sum(physmon_lens)

                if max_monomial_length > 0:
                    max_total_mon_length = max_monomial_length

                party_frequencies = []
                for pfreq in itertools.product(*[range(physmon_lens[party]+1) for party in range(self.nr_parties)]):
                    if sum(pfreq) <= max_total_mon_length:
                        party_frequencies.append(list(reversed(pfreq)))
                party_frequencies = sorted(party_frequencies, key=sum)

                physical_monomials = []
                for freqs in party_frequencies:
                    if freqs == [0]*self.nr_parties:
                        physical_monomials.append([0])
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
                                #template_of_len_party_0 = template_physmon_all_lens[freq-1]
                                #with_correct_party_idx = physmon_change_party_in_template(template_of_len_party_0, party)
                                physmons = phys_mon_1_party_of_given_len(self.InflationProblem.hypergraph,
                                                                         self.InflationProblem.inflation_level_per_source,
                                                                         party, freq,
                                                                         self.InflationProblem.settings_per_party,
                                                                         self.InflationProblem.outcomes_per_party,
                                                                         self.InflationProblem.names)
                                physmons_per_party_per_length.append(physmons)

                        for mon_tuple in itertools.product(*physmons_per_party_per_length):
                            concatenated = mon_tuple[0]
                            for i in range(1, len(mon_tuple)):
                                concatenated = np.concatenate((concatenated, mon_tuple[i]), axis=0)

                            physical_monomials.append(concatenated.tolist())

                columns = physical_monomials
            else:
                raise Exception('I have not understood the format of the '
                                + 'column specification')
        else:
            raise Exception('I have not understood the format of the '
                            + 'column specification')

        columns_symbolical = from_coord_to_sym(columns,
                                               self.names,
                                               self.InflationProblem.nr_sources,
                                               self.measurements)

        if return_columns_numerical:
            return columns_symbolical, columns
        else:
            return columns_symbolical

    def _build_cols_from_col_specs(self, col_specs: List[List]) -> None:
        """This builds the generating set for the moment matrix taking as input
        a block specified only the number of parties, and the party labels.

        For example, with col_specs=[[], [0], [2], [0, 2]] as input, we
        generate the generating set S={1, A_{ijk}_xa, C_{lmn}_zc,
        A_{i'j'k'}_x'a' * C{l'm'n'}_{z'c'}} where i,j,k,l,m,n,i',j',k',l',m',n'
        represent all possible inflation copies indices compatible with the
        network structure, and x,a,z,c,x',a',z',c' are all possible input
        and output indices compatible with the cardinalities. As further
        examples, NPA level 2 for 3 parties is built from
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

        if self.substitutions == {}:
            if all([len(block) == len(np.unique(block)) for block in col_specs]):
                # TODO maybe do things without sympy substitutions,
                # but numba functions??
                warn("You have not input substitution rules to the " +
                     "generation of columns, but it is OK because you " +
                     "are using local level 1")
            else:
                raise Exception("You must input substitution rules for columns "
                                + "to be generated properly")
        else:
            # party_structure is something like:
            # only the first element is special, and it determines whether we include 1 or not
            # TODO change with numba functions
            res = []
            symbols = []
            for block in col_specs:
                if block == []:
                    res.append([0])
                    symbols.append(1)
                else:
                    meas_ops = []
                    for party in block:
                        meas_ops.append(flatten(self.measurements[party]))
                    for monomial_factors in itertools.product(*meas_ops):
                        monomial = apply_substitutions(
                            np.prod(monomial_factors), self.substitutions)
                        # The commutation rules in self.substitutions are not
                        # enough in some occasions when the monomial can be
                        # factorized. An example is (all indices are inflation
                        # indices) A13A33A22 and A22A13A33. Both are the same
                        # because they are composed of two disconnected objects,
                        # but cannot be brought one to another via two-body
                        # substitution rules. The solution below is to decompose
                        # in disconnected components, order them canonically,
                        # and recombine them.
                        factor_list = monomial.as_ordered_factors()
                        num_to_symb = {tuple(to_numbers(str(factor),
                                                        self.names)[0]): factor
                                                      for factor in factor_list}
                        reordered = np.concatenate(
                                        factorize_monomial(
                                            to_numbers(str(monomial),
                                                       self.names)))
                        # Recombine sorting by party
                        reordered = np.vstack(sorted(reordered,
                                                     key=lambda x: x[0]))
                        monomial = np.prod([num_to_symb[tuple(factor)]
                                            for factor in reordered])
                        mon_length = len(str(monomial).split('*'))
                        if monomial not in symbols and mon_length == len(block):
                            symbols.append(monomial)
                            if monomial == 1:
                                coords = [0]
                            else:
                                coords = []
                                for factor in monomial.as_coeff_mul()[1]:
                                    coords.append(*to_numbers(factor, self.names))
                            res.append(coords)
            sortd = sorted(res, key=len)
            return [np.array(mon, dtype=np.uint8) for mon in sortd]

    def _generate_parties(self):
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

        hypergraph = self.InflationProblem.hypergraph
        settings   = self.InflationProblem.settings_per_party
        outcomes   = self.InflationProblem.outcomes_per_party
        inflation_level_per_source = self.InflationProblem.inflation_level_per_source
        commuting = self.commuting

        assert len(settings) == len(
            outcomes), 'There\'s a different number of settings and outcomes'
        assert len(
            settings) == hypergraph.shape[1], 'The hypergraph does not have as many columns as parties'
        substitutions = {}
        measurements = []
        # [chr(i) for i in range(ord('A'), ord('A') + hypergraph.shape[1])]
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

            # Generate measurements for every combination of indices. The two
            # outcome case is handled separately for having the same format always
            for indices in all_indices:
                if outs == 2:
                    if not commuting:
                        meas = generate_noncommuting_measurements([outs + 1 for _ in range(ins)],
                                                                  party + '_' + '_'.join(indices))
                    else:
                        meas = generate_commuting_measurements([outs + 1 for _ in range(ins)],
                                                               party + '_' + '_'.join(indices))
                    for i in range(ins):
                        meas[i].pop(-1)
                else:
                    if not commuting:
                        meas = generate_noncommuting_measurements([outs for _ in range(ins)],
                                                                  party + '_' + '_'.join(indices))
                    else:
                        meas = generate_commuting_measurements([outs for _ in range(ins)],
                                                               party + '_' + '_'.join(indices))

                if commuting:
                    subs = projective_measurement_constraints(meas)
                    substitutions = {**substitutions, **subs}
                party_meas.append(meas)
            measurements.append(party_meas)

        if not commuting:
            substitutions = {}
            for party in measurements:
                # Idempotency
                substitutions = {**substitutions,
                                 **{op**2: op for op in flatten(party)}}
                # Orthogonality
                for inf_copy in party:
                    for measurement in inf_copy:
                        for out1 in measurement:
                            for out2 in measurement:
                                if out1 == out2:
                                    substitutions[out1*out2] = out1
                                else:
                                    substitutions[out1*out2] = 0
            # Commutation of different parties
            for i in range(len(parties)):
                for j in range(len(parties)):
                    if i > j:
                        for op1 in flatten(measurements[i]):
                            for op2 in flatten(measurements[j]):
                                substitutions[op1*op2] = op2*op1
            # Operators for a same party with non-overlapping copy indices commute
            for party, inf_measurements in enumerate(measurements):
                sources = hypergraph[:, party].astype(bool)
                inflation_indices = [np.compress(sources,
                                                 str(flatten(inf_copy)[0]).split('_')[1:-2]).astype(int)
                                     for inf_copy in inf_measurements]
                for ii, first_copy in enumerate(inf_measurements):
                    for jj, second_copy in enumerate(inf_measurements):
                        if (jj > ii) and (all(inflation_indices[ii] != inflation_indices[jj])):
                            for op1, op2 in itertools.product(flatten(first_copy), flatten(second_copy)):
                                substitutions[op2 * op1] = op1 * op2

        return measurements, substitutions, parties

    def _build_momentmatrix(self) -> Tuple[np.ndarray, Dict]:
        """Generate the moment matrix.
        """

        _cols = [np.array(col, dtype=np.uint8)
                    for col in self.generating_monomials]
        problem_arr, canonical_mon_to_idx_dict = calculate_momentmatrix(_cols,
                                                                        verbose=self.verbose,
                                                                        commuting=self.commuting)

        idx_to_canonical_mon_dict = {idx: mon for (mon, idx) in canonical_mon_to_idx_dict.items() if idx>=2}
        # # Remove duplicates in vardic that have the same index
        # vardic_clean = {}
        # for mon, idx in canonical_mon_to_idx_dict.items():
        #     if idx not in vardic_clean:
        #         vardic_clean[idx] = mon
        # monomials_list = np.array(
        #     list(vardic_clean.items()), dtype=str).astype(object)
        # monomials_list = monomials_list[1:]  # Remove the '1': ' ' row

        # TODO change from dense to sparse !! Else useless, but this requires adapting code
        problem_arr = problem_arr.todense()

        return problem_arr, idx_to_canonical_mon_dict

    def _calculate_inflation_symmetries(self) -> List[List]:
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


        inflevel  = self.InflationProblem.inflation_level_per_source
        n_sources = len(inflevel)
        # Start with the identity permutation
        inflation_symmetries = [list(range(len(self.generating_monomials)))]

        # TODO do this function without relying on symbolic substitutions!!
        flatmeas  = np.array(flatten(self.measurements))
        measnames = np.array([str(meas) for meas in flatmeas])

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
                                                      self.substitutions,
                                                      flatmeas,
                                                      measnames,
                                                      self.names)
            # Bring columns to the canonical form using commutations of
            # connected components
            for idx in range(len(permuted_cols_ind)):
                if len(permuted_cols_ind[idx].shape) > 1:
                    permuted_cols_ind[idx] = np.vstack(
                                 sorted(np.concatenate(factorize_monomial(
                                                          permuted_cols_ind[idx]
                                                                          )),
                                        key=lambda x: x[0])
                                                       )
            list_permuted = from_numbers_to_flat_tuples(permuted_cols_ind)
            try:
                total_perm = find_permutation(list_permuted, list_original)
                inflation_symmetries.append(total_perm)
            except:
                if self.verbose > 0:
                    warn("The generating set is not closed under source swaps."+
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
                                    inflation_symmetries: List[List[int]]
                                    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Applies the inflation symmetries to the moment matrix.

        Parameters
        ----------
        momentmatrix : np.ndarray
            The moment matrix.
        monomials_list : np.ndarray
            The list of monomials as List[Tuple[int, ArrayMonomial]]
        inflation_symmetries : List[List[int]]


        Returns
        -------
        Tuple[np.ndarray, Dict[int, int], np.ndarray]
            The symmetrized moment matrix, the orbits as a dictionary
            and the symmetrized monomials list.
        """

        symmetric_arr = momentmatrix.copy()
        if len(symmetric_arr.shape) == 2:
            # TODO This is inelegant, remove the index.
            #  Only here for compatibility reasons
            aux = np.zeros((symmetric_arr.shape[0], symmetric_arr.shape[1], 2))
            aux[:, :, 0] = symmetric_arr
            symmetric_arr = aux.astype(int)

        indices_to_delete = []
        # the +2 is to include 0:0 and 1:1
        # orbits = {i: i for i in range(2+len(monomials_list))}
        # orbits = {i: i for i in np.unique(sdp.problem_arr.flat)}
        orbits = np.unique(symmetric_arr.flat)
        for permutation in tqdm(inflation_symmetries,
                                disable=not self.verbose,
                                desc="Applying symmetries          "):
            for i, ip in enumerate(permutation):
                for j, jp in enumerate(permutation):
                    if symmetric_arr[i, j, 0] < symmetric_arr[ip, jp, 0]:
                        indices_to_delete.append(int(symmetric_arr[ip, jp, 0]))
                        orbits[symmetric_arr[ip, jp, 0]
                               ] = symmetric_arr[i, j, 0]
                        symmetric_arr[ip, jp, :] = symmetric_arr[i, j, :]

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
                    warn("Your generating set might not have enough" +
                         "elements to fully impose inflation symmetries.")
            orbits[key] = val

        old_representative_indices, new_indices, unsym_idx_to_sym_idx = np.unique(orbits,
                                                                                  return_index=True,
                                                                                  return_inverse=True)
        ###We need the check if the special indices 0 and 1 are IN orbits at all. IF not, we will need to shift the new
        ###indices up a bit.
        if not 1 in orbits:
            new_indices[new_indices>0] = new_indices+1
            unsym_idx_to_sym_idx[unsym_idx_to_sym_idx>0] = unsym_idx_to_sym_idx+1
        if not 0 in orbits:
            new_indices = new_indices + 1
            unsym_idx_to_sym_idx = unsym_idx_to_sym_idx + 1

        symmetrized_momentmatrix = unsym_idx_to_sym_idx.take(momentmatrix)
        symidx_to_canonical_mon_dict = {new_idx: unsymidx_to_canonical_mon_dict[old_idx] for old_idx, new_idx in zip(
            old_representative_indices,
            new_indices) if old_idx>=2}


        # Remove from monomials_list all those that have disappeared. The -2 is
        # because we are encoding 0 and 1 in two variables that we do not use
        # remaining_variables = (set(range(len(monomials_list))) -
        #                        set(np.array(indices_to_delete)-2))
        # remaining_monomials = monomials_list[sorted(list(remaining_variables))]

        # return symmetric_arr.astype(int), orbits, remaining_monomials
        return symmetrized_momentmatrix, orbits, symidx_to_canonical_mon_dict

    def _factorize_monomials(self,
                             monomials: np.ndarray,
                             combine_unknowns: bool=True
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """Splits the monomials into factors according to the supports of the
        operators.

        Parameters
        ----------
        monomials : np.ndarray
            List of unfactorised monomials.
        combine_unknowns : bool
            Whether combining the unknown monomials into a single one.
            Default True.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The factorised monomials reordered according to know, semiknown
            and unknown moments and also the corresponding unfactorised
            monomials, also reordered.
        """
        monomials_factors = factorize_monomials(
                                monomialset_name2num(monomials, self.names),
                                                verbose=self.verbose)
        monomials_factors_names = monomialset_num2name(
            monomials_factors, self.names)
        monomials_factors_knowable, factors_are_knowable = \
                           self.label_knowable_and_unknowable(monomials_factors)

        # Some counting
        self._n_known           = np.sum(monomials_factors_knowable[:, 1] == 'Yes')
        self._n_something_known = np.sum(monomials_factors_knowable[:, 1] != 'No')
        self._n_unknown         = np.sum(monomials_factors_knowable[:, 1] == 'No')

        # Recombine multiple unknowable variables into one, and reorder the
        # factors so the unknowable is always last
        multiple_unknowable = []
        for idx, [_, are_knowable] in enumerate(factors_are_knowable):
            if len(are_knowable) >= 2:
                are_knowable = np.array(are_knowable)
                if (np.count_nonzero(1-are_knowable) > 1) and combine_unknowns:
                    # Combine unknowables and reorder
                    unknowable_factors = np.concatenate(
                        [factor
                           for i, factor in enumerate(monomials_factors[idx][1])
                            if not are_knowable[i]], axis=0)
                    unknowable_factors_ordered = \
                           unknowable_factors[unknowable_factors[:,0].argsort()]
                    # Ugly hack. np.concatenate and np.append were failing
                    monomials_factors[idx][1] = \
                        ([factor
                           for i, factor in enumerate(monomials_factors[idx][1])
                            if are_knowable[i]]
                        + [unknowable_factors_ordered])
                else:
                    # Just reorder
                    monomials_factors[idx][1] = \
                        ([factor
                           for i, factor in enumerate(monomials_factors[idx][1])
                         if are_knowable[i]]
                        +
                        [factor
                           for i, factor in enumerate(monomials_factors[idx][1])
                          if not are_knowable[i]])

        # Reorder according to known, semiknown and unknown.
        monomials_factors_reordered = np.concatenate(
            [monomials_factors[monomials_factors_knowable[:, 1] == 'Yes'],
             monomials_factors[monomials_factors_knowable[:, 1] == 'Semi'],
             monomials_factors[monomials_factors_knowable[:, 1] == 'No']]
                                                            )

        monomials_unfactorised_reordered = np.concatenate(
            [monomials[monomials_factors_knowable[:, 1] == 'Yes'],
             monomials[monomials_factors_knowable[:, 1] == 'Semi'],
             monomials[monomials_factors_knowable[:, 1] == 'No']]
                                                            )
        monomials_unfactorised_reordered = monomials_unfactorised_reordered.astype(object)
        monomials_unfactorised_reordered[:, 0] = monomials_unfactorised_reordered[:, 0].astype(int)

        return monomials_factors_reordered, monomials_unfactorised_reordered

    def _find_positive_monomials(self, monomials_factors: np.ndarray,
                                       sandwich_positivity=True):
        ispositive = np.empty_like(monomials_factors)
        ispositive[:, 0] = monomials_factors[:, 0]
        ispositive[:, 1] = False
        ispositive[:self._n_known, 1] = True    # Knowable moments are physical
        for i, row in enumerate(monomials_factors[self._n_known:]):
            factors = row[1]
            factor_is_positive = []
            for factor in factors:
                isphysical = is_physical(factor,
                                         sandwich_positivity=sandwich_positivity)
                factor_is_positive.append(isphysical)
            if all(factor_is_positive):
                ispositive[i+self._n_known, 1] = True
        return monomials_factors[ispositive[:, 1].astype(bool), 0]

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
        #Parties start at #1 in our numpy vector notation.
        scenario_subhypergraph = self.hypergraph[:, parties - 1]
        monomial_sources = np.asarray(monomial)[:, 1:-2].T
        monomial_hypergraph = monomial_sources.copy()
        monomial_hypergraph[np.nonzero(monomial_hypergraph)] = 1
        return all([source in scenario_subhypergraph.tolist()
                    for source in monomial_hypergraph.tolist()])

    def label_knowable_and_unknowable(self,
                                      monomials_factors_input: np.ndarray
                                      ) -> np.ndarray:
        """Given the list of monomials factorised, it labels each
        monomial into knowable, semiknowable and unknowable.

        Parameters
        ----------
        monomials_factors_input : np.ndarray
            Ndarray of factorised monomials. Each row encodes the integer
            representation and the factors of the monomial.


        Returns
        -------
        np.ndarray
            Array of the same size as the input, with the labels of each monomial.
        """
        monomials_factors_knowable = np.empty_like(monomials_factors_input)
        monomials_factors_knowable[:, 0] = monomials_factors_input[:, 0]
        factors_are_knowable       = np.empty_like(monomials_factors_input)
        factors_are_knowable[:, 0] = monomials_factors_input[:, 0]
        for idx, [_, monomial_factors] in enumerate(tqdm(monomials_factors_input, disable=True)):
            factors_known_list = [self.atomic_knowable_q(factor) for factor in monomial_factors]
            factors_are_knowable[idx][1] = factors_known_list
            if all(factors_known_list):
                knowable = 'Yes'
            elif any(factors_known_list):
                knowable = 'Semi'
            else:
                knowable = 'No'
            monomials_factors_knowable[idx][1] = knowable
        return monomials_factors_knowable, factors_are_knowable

if __name__ == "__main__":
    sdp = InflationSDP(InflationProblem(dag={'U_AB': ['A','B'],
                                       'U_AC': ['A','C'],
                                       'U_AD': ['A','D'],
                                       'C': ['D'],
                                       'A': ['B', 'C', 'D']},
                                  outcomes_per_party=(2, 2, 2, 2),
                                  settings_per_party=(1, 1, 1, 1),
                                  inflation_level_per_source=(1, 1, 1),
                                  names=('A', 'B', 'C', 'D'),
                                  verbose=2),
                       commuting=False,
                       verbose=2)
    sdp.generate_relaxation('local1')

    # cutInflation = InflationProblem({"lambda": ["a", "b"],
    #                                  "mu": ["b", "c"],
    #                                  "sigma": ["a", "c"]},
    #                                  names=['a', 'b', 'c'],
    #                                  outcomes_per_party=[2, 2, 2],
    #                                  settings_per_party=[1, 1, 1],
    #                                  inflation_level_per_source=[2, 1, 1])
    # sdp = InflationSDP(cutInflation)
    # sdp.generate_relaxation('local1')

    # print(sdp.unsymmetrized_mm_idxs)
    # print(list(sdp.unsymidx_to_canonical_mon_dict.items()))
    # print(sdp.momentmatrix)
    # print(len(sdp.symidx_to_canonical_mon_dict))
    # print(len(sdp.symidx_to_Monomials_dict))
    #
    # for k, mon in sdp.symidx_to_Monomials_dict.items():
    #
    #     print(f"{k} := {mon}, {mon.knowability_status}")