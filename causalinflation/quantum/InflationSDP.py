import copy
import itertools
import numpy as np
import sympy as sp

from causalinflation import InflationProblem
from causalinflation.quantum.general_tools import (to_name, to_representative,
                                            to_numbers, mul,
                                            transform_vars_to_symb,
                                            substitute_variable_values_in_monlist,
                                            substitute_sym_with_numbers,
                                            string2prob,
                                            phys_mon_1_party_of_given_len,
                                            is_physical,
                                            label_knowable_and_unknowable,
                                            monomialset_name2num,
                                            monomialset_num2name,
                                            factorize_monomials,
                                            factorize_monomial,
                                            find_permutation,
                                            apply_source_permutation_coord_input,
                                            from_numbers_to_flat_tuples,
                                            generate_noncommuting_measurements,
                                            flatten,
                                            from_coord_to_sym,
                                            clean_coefficients,
                                            compute_numeric_value)
from causalinflation.quantum.fast_npa import (calculate_momentmatrix,
                                              calculate_momentmatrix_commuting,
                                              to_canonical,
                                              dot_mon, mon_is_zero, mon_lexsorted,
                                              remove_projector_squares,
from causalinflation.quantum.sdp_utils import (solveSDP_MosekFUSION,
                                                solveSDP_MosekFUSION2)
from causalinflation.quantum.writer_utils import (write_to_csv, write_to_mat,
                                                  write_to_sdpa)
from typing import List, Dict, Union, Tuple
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
        problem_arr, self._monomials_list_all, self._mon_string2int = \
                                                      self._build_momentmatrix()
        if self.verbose > 1:
            print("Number of variables before symmetrization:",
                  len(self._monomials_list_all))

        # Calculate the inflation symmetries.
        inflation_symmetries = self._calculate_inflation_symmetries()

        # Apply the inflation symmetries to the moment matrix.
        # TODO: make it so that after this 'monomials_list' is no longer
        # needed and can be safely deleted. Currently this list can get big
        # which just occupies needless memory. Also currently its only used
        # for the objective function, because the user might input some
        # monomial that is not in the form of the ones found in
        # 'remaining_monomials'. The correct thing to do is to use a function
        # to bring the monomials in the user-inputted objective function
        # to a canonoical form! But this is not implemented yet.
        symmetric_arr, orbits, remaining_monomials \
                    = self._apply_inflation_symmetries(problem_arr,
                                                       self._monomials_list_all,
                                                       inflation_symmetries)
        # Associate the names of all copies to the same variable
        for key, val in self._mon_string2int.items():
            self._mon_string2int[key] = orbits[val]

        # Bring remaining monomials to a representative form, and add the
        # corresponding identifications to the dictionary
        for idx, [var, mon] in enumerate(tqdm(
                                       monomialset_name2num(remaining_monomials,
                                                            self.names),
                                              disable=not self.verbose,
                                          desc="Computing canonical forms   ")):
            canonical = to_name(to_representative(np.array(mon),
                                                  self.inflation_levels,
                                                  self.commuting),
                                self.names)
            remaining_monomials[idx][1] = canonical
            self._mon_string2int[canonical] = var

        monomials_factors, monomials_unfactorised_reordered \
                            = self._factorize_monomials(remaining_monomials,
                                                        combine_unknowns=True)

        self.monomials_list = monomials_unfactorised_reordered

        # Reassign the integer variable names to ordered from 1 to N
        variable_dict = {**{0: 0, 1: 1},
                         **dict(zip(self.monomials_list[:, 0],
                              range(2, self.monomials_list.shape[0] + 2)))}

        self._var2repr = {key: variable_dict[val]
                              for key, val in orbits.items()}
        self._mon2indx = {key: variable_dict[val]
                              for key, val in self._mon_string2int.items()}

        # Change objects to new variables
        self.momentmatrix = symmetric_arr[:, :, 0].astype(int)
        for i, row in enumerate(tqdm(self.momentmatrix,
                                     disable=not self.verbose,
                                     desc="Reassigning moment matrix indices")):
            for j, col in enumerate(row):
                self.momentmatrix[i, j] = self._var2repr[col]

        for idx in range(len(self.monomials_list)):
            self.monomials_list[idx, 0] = self._var2repr[self.monomials_list[idx, 0]]
            monomials_factors[idx, 0]   = self._var2repr[monomials_factors[idx, 0]]

        # Find all the positive monomials
        if self.commuting:
            self.physical_monomials = monomials_factors[:,0]
        else:
            self.physical_monomials = self._find_positive_monomials(
                monomials_factors, sandwich_positivity=True)

        if self.verbose > 0:
            print("Number of known, semi-known and unknown variables =",
                    self._n_known, self._n_something_known-self._n_known,
                    self._n_unknown)
            print("Number of positive unknown variables =",
                  len(self.physical_monomials) - self._n_known)
            if self.verbose > 1:
                print("Positive variables:",
                      [self.monomials_list[phys-2]
                                           for phys in self.physical_monomials])

        # Store the variables that will be relevant when setting a distribution
        # and into which variables they factorize
        monomials_factors_vars = np.empty_like(monomials_factors).tolist()
        for idx, [var, factors] in enumerate(monomials_factors):
            monomials_factors_vars[idx][0] = var
            factor_variables = []
            for factor in factors:
                try:
                    factor_variables.append(self._mon2indx[
                               to_name(to_representative(np.array(factor),
                                                         self.inflation_levels,
                                                         self.commuting),
                                       self.names)])
                except KeyError:
                    # If the unknown variable doesn't appear anywhere else, add
                    # it to the list
                    self._n_unknown += 1
                    var_idx = self._n_something_known + self._n_unknown
                    self._mon2indx[to_name(to_representative(np.array(factor),
                                                          self.inflation_levels,
                                                             self.commuting),
                            self.names)] = var_idx
                    factor_variables.append(var_idx)
            monomials_factors_vars[idx][1] = factor_variables
        self.semiknowable_atoms = monomials_factors_vars[:self._n_something_known]
        # Define trivial arrays for distribution and objective
        self.known_moments      = {0: 0., 1: 1.}
        self.semiknown_moments  = {}
        self.objective          = 0.
        self._objective_as_dict = {1: 0.}
        
        # For the bounds, monomials should be hashed in the same way as 
        # self.known_moments, self._objective_as_dict, etc.
        self.moment_linear_equalities = []
        self.moment_linear_inequalities = []
        self.moment_lowerbounds = {physical: 0 for physical 
                                                    in self.physical_monomials}
        self.moment_upperbounds = {}

        

    def set_distribution(self,
                         prob_array: np.ndarray,
                         use_lpi_constraints: bool = False) -> None:
        """Set numerically the knowable moments and semiknowable moments according
        to the probability distribution specified. If p is None, or the user
        doesn't pass any argument to set_distribution, then this is understood
        as a request to delete information about past distributions. If p containts
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

        self.use_lpi_constraints = use_lpi_constraints

        if (len(self._objective_as_dict) > 1) and self.use_lpi_constraints:
            warn("You have an objective function set. Be aware that imposing " +
                 "linearized polynomial constraints will constrain the " +
                 "optimization to distributions with fixed marginals.")

        atomic_knowable_variables = [entry
                for idx, entry in enumerate(self.monomials_list[:self._n_known])
                                   if len(self.semiknowable_atoms[idx][1]) == 1]
        atomic_numerical_values = {var[0]: compute_numeric_value(var[1],
                                                                 prob_array,
                                                    self.InflationProblem.names)
                                   for var in atomic_knowable_variables}
        self.set_values(atomic_numerical_values,
                        self.use_lpi_constraints,
                        only_specified_values=False)


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
            objective = sp.expand(objective)
            # Build string-to-variable dictionary
            string2int_dict = {**{'0': '0', '1': '1'},
                               **dict(self._monomials_list_all[:, ::-1])}
            # Express objective in terms of representatives
            symmetrized_objective = {1: 0.}
            for monomial, coeff in objective.as_coefficients_dict().items():
                try:
                    monomial_variable = int(string2int_dict[str(monomial)])
                except KeyError:
                    Exception("The objective function contains variables that" +
                              " are not found in the generated relaxation." +
                              " Consider adding more monomials to the" +
                              " generating set.")
                repr = self._var2repr[monomial_variable]
                # If the objective contains a known value add it to the constant
                if repr in self.known_moments:
                    warn("Be aware that you have variables in the objective " +
                         "that are also known moments fixed by your distribution.")
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
                   values: Dict[Union[sp.core.symbol.Symbol, sp.core.mul.Mul,
                                      int, str], float],
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
            elif type(key) in [str, sp.core.mul.Mul, sp.core.symbol.Symbol]:
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

    def certificate_as_string(self,
                               clean: bool=False,
                               chop_tol: float=1e-10,
                               round_decimals: int=3) -> sp.core.add.Add:
        """Give the certificate as a string with the notation of the operators
        in the moment matrix.

        Parameters
        ----------
        clean : bool, optional
            If true, eliminate all coefficients that are smaller
            than 'chop_tol' and round to the number of decimals specified
            `round_decimals`, by default False.
        chop_tol : float, optional
            Coefficients in the dual certificate smaller in absolute value are
            set to zero, by default 1e-8.
        round_decimals : int, optional
            Coefficients that are not set to zero are rounded to the number
            of decimals specified, by default 3.

        Returns
        -------
        sympy.core.add.Add
            The certificate in terms of symbols representing the monomials in
            the moment matrix. The certificate of infeasibility is cert > 0.
        """
        try:
            dual = self.solution_object['dual_certificate']
        except AttributeError:
            raise Exception("For extracting a certificate you need to solve " +
                            "a problem. Call 'InflationSDP.solve()' first")
        if len(self.semiknown_moments) > 0:
            warn("Beware that, because the problem contains linearized " +
                 "polynomial constraints, the certificate is not guaranteed " +
                 "to apply to other distributions")

        if clean and not np.allclose(list(dual.values()), 0.):
            dual = clean_coefficients(dual, chop_tol, round_decimals)

        vars_to_names = {**{0: 0., 1: 1.}, **dict(self.monomials_list)}
        cert = str(dual[1])
        for var, coeff in dual.items():
            if var > 1:
                cert += "+" if coeff > 0 else "-"
                cert += f"{abs(coeff)}*{vars_to_names[var]}"
        cert += " >= 0"
        return cert

    def certificate_as_probs(self,
                             clean: bool=False,
                             chop_tol: float=1e-10,
                             round_decimals: int=3) -> sp.core.add.Add:
        """Give certificate as symbolic sum of probabilities. The certificate
        of incompatibility is cert >= 0.

        Parameters
        ----------
        clean : bool, optional
            If true, eliminate all coefficients that are smaller
            than 'chop_tol' and round to the number of decimals specified
            `round_decimals`, by default False.
        chop_tol : float, optional
            Coefficients in the dual certificate smaller in absolute value are
            set to zero, by default 1e-8.
        round_decimals : int, optional
            Coefficients that are not set to zero are rounded to the number
            of decimals specified, by default 3.

        Returns
        -------
        sympy.core.add.Add
            The expression of the certificate in terms or probabilities and
            marginals. The certificate of incompatibility is cert >= 0.
        """
        try:
            dual = self.solution_object['dual_certificate']
        except AttributeError:
            raise Exception("For extracting a certificate you need to solve " +
                            "a problem. Call 'InflationSDP.solve()' first")
        if len(self.semiknown_moments) > 0:
            warn("Beware that, because the problem contains linearized " +
                 "polynomial constraints, the certificate is not guaranteed " +
                 "to apply to other distributions")

        if clean and not np.allclose(list(dual.values()), 0.):
            dual = clean_coefficients(dual, chop_tol, round_decimals)

        vars_to_factors = dict(self.semiknowable_atoms)
        vars_to_names   = {**{0: 0., 1: 1.}, **dict(self.monomials_list)}
        cert = dual[1]
        for var, coeff in dual.items():
            if var > 1:
                try:
                    factors = 1
                    for factor in vars_to_factors[var]:
                        prob = string2prob(vars_to_names[factor],
                                           self.nr_parties)
                        factors *= prob
                    cert += coeff * factors
                except KeyError:
                    cert += coeff * sp.Symbol(vars_to_names[var])
        return cert

    def certificate_as_objective(self, clean: bool=False,
                                 chop_tol: float=1e-10,
                                 round_decimals: int=3) -> sp.core.add.Add:
        """Give certificate as symbolic sum of operators that can be used
        as an objective function to optimse.

        Parameters
        ----------
        clean : bool, optional
            If true, eliminate all coefficients that are smaller
            than 'chop_tol', normalise and round to the number of decimals
            specified `round_decimals`, by default False.
        chop_tol : float, optional
            Coefficients in the dual certificate smaller in absolute value are
            set to zero, by default 1e-8.
        round_decimals : int, optional
            Coefficients that are not set to zero are rounded to the number
            of decimals specified, by default 3.

        Returns
        -------
        sympy.core.add.Add
            The certificate as an objective function.
        """
        try:
            dual = self.solution_object['dual_certificate']
        except AttributeError:
            raise Exception("For extracting a certificate you need to solve " +
                            "a problem. Call 'InflationSDP.solve()' first")
        if len(self.semiknown_moments) > 0:
            warn("Beware that, because the problem contains linearized " +
                 "polynomial constraints, the certificate is not guaranteed " +
                 "to apply to other distributions")

        if clean and not np.allclose(list(dual.values()), 0.):
            dual = clean_coefficients(dual, chop_tol, round_decimals)

        cert = dual[1]
        for var, coeff in dual.items():
            if var > 1:
                # Find position of the first appearance of the variable
                i, j = np.array(np.where(self.momentmatrix == var))[:,0]
                m1 = self.generating_monomials[i] if i > 0 else np.array([[0]])
                m2 = self.generating_monomials[j] if j > 0 else np.array([[0]])

                # Create the monomial
                monom  = dot_mon(m1, m2)
                if self.commuting:
                    canonical = remove_projector_squares(mon_lexsorted(monom))
                    if mon_is_zero(canonical):
                        canonical = 0
                else:
                    canonical = to_canonical(monom)
                # Generate symbolic representation
                symb = 1
                for element in canonical:
                    party  = element[0] - 1                # Name indices from 0
                    inf    = np.array(element[1:-2]) - 1   # Name indices from 0
                    input  = element[-2]
                    output = element[-1]
                    inf[inf < 0] = 0             # Negative indices are not used
                    inf_idx = self.inflation_levels@inf
                    symb   *= self.measurements[party][inf_idx][input][output]
                cert += coeff*symb
        return cert

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
        elif extension == 'csv':
            write_to_csv(self, filename)
        elif extension == 'mat':
            write_to_mat(self, filename)
        else:
            raise Exception('File format not supported. Please choose between' +
                            ' the extensions .csv, .dat-s and .mat.')

    def clear_known_values(self) -> None:
        self.known_moments     = {0: 0., 1: 1.}
        self.semiknown_moments = {}

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
                        physical_monomials.append(np.array([0]))
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
                            concatenated = np.concatenate(mon_tuple, axis=0)
                            # Bring to canonical form
                            monomial = np.vstack(
                                           sorted(
                                               np.concatenate(
                                                   factorize_monomial(concatenated)),
                                                  key=lambda x: x[0]))
                            physical_monomials.append(monomial)

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

        res = []
        allvars = set()
        for block in col_specs:
            if block == []:
                res.append(np.array([0]))
                allvars.add('1')
            else:
                meas_ops = []
                for party in block:
                    meas_ops.append(flatten(self.measurements[party]))
                for monomial_factors in itertools.product(*meas_ops):
                    mon = np.array([to_numbers(op, self.names)[0]
                                    for op in monomial_factors])
                    if self.commuting:
                        canon = remove_projector_squares(mon_lexsorted(mon))
                        if mon_is_zero(canon):
                            canon = 0
                    else:
                        canon = to_canonical(mon)
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
                                    res.append(np.array([0]))
                                else:
                                    res.append(canon)

        sortd = sorted(res, key=len)
        return sortd

    def _generate_parties(self):
        # TODO: change name to generate_measurements
        """Generates all the party operators and substitution rules in an
        quantum inflation setup on a network given by a hypergraph. It uses
        information stored in the `InflationSDP` instance.

        It stores in `self.measurements` a list of lists of measurement
        operators indexed as self.measurements[p][c][i][o] for party p,
        copies c, input i, output o.

        It stores in `self.substitutions` a dictionary of substitution rules
        containing commutation, orthogonality and square constraints.
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
                meas = generate_noncommuting_measurements(
                                                 [outs - 1 for _ in range(ins)],   # -1 because of normalization
                                                 party + '_' + '_'.join(indices)
                                                          )
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
                        substitutions[m1*m2] = m2*m1
                    elif str(m1) < str(m2):
                        substitutions[m2*m1] = m1*m2
                    else:
                        pass
        else:
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
                                                 str(flatten(inf_copy)[0]
                                                     ).split('_')[1:-2]
                                                 ).astype(int)
                                     for inf_copy in inf_measurements]
                for ii, first_copy in enumerate(inf_measurements):
                    for jj, second_copy in enumerate(inf_measurements):
                        if ((jj > ii) and
                            (all(inflation_indices[ii] != inflation_indices[jj]))):
                            for op1, op2 in itertools.product(flatten(first_copy),
                                                              flatten(second_copy)):
                                substitutions[op2 * op1] = op1 * op2
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

        return measurements, substitutions, parties

    def _build_momentmatrix(self) -> None:
        """Generate the moment matrix.
        """

        _cols = [np.array(col, dtype=np.uint8)
                    for col in self.generating_monomials]
        if not self.commuting:
            problem_arr, vardic = \
                            calculate_momentmatrix(_cols,
                                                np.array(self.names),
                                                verbose=self.verbose)
        else:
            problem_arr, vardic = \
                            calculate_momentmatrix_commuting(_cols,
                                                    np.array(self.names),
                                                    verbose=self.verbose)

        # Remove duplicates in vardic that have the same index
        vardic_clean = {}
        for k, v in vardic.items():
            if v not in vardic_clean:
                vardic_clean[v] = k
        monomials_list = np.array(
            list(vardic_clean.items()), dtype=str).astype(object)
        monomials_list = monomials_list[1:]  # Remove the '1': ' ' row

        # TODO change from dense to sparse !! Else useless, but this requires adapting code
        problem_arr = problem_arr.todense()

        return problem_arr, monomials_list, vardic

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
        inflation_symmetries = []

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
                                                      self.commuting)
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

        return inflation_symmetries

    def _apply_inflation_symmetries(self,
                                    momentmatrix: np.ndarray,
                                    monomials_list: np.ndarray,
                                    inflation_symmetries: List[List[int]]
                                    ) -> Tuple[np.ndarray, Dict[int, int], np.ndarray]:
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
        orbits = {i: i for i in range(2+len(monomials_list))}
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
        for key, val in orbits.items():
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

        # Remove from monomials_list all those that have disappeared. The -2 is
        # because we are encoding 0 and 1 in two variables that we do not use
        remaining_variables = (set(range(len(monomials_list))) -
                               set(np.array(indices_to_delete)-2))
        remaining_monomials = monomials_list[sorted(list(remaining_variables))]

        return symmetric_arr.astype(int), orbits, remaining_monomials

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
        is_knowable, factors_are_knowable = label_knowable_and_unknowable(
                            monomials_factors, self.InflationProblem.hypergraph)

        # Some counting
        self._n_known           = np.sum(is_knowable[:, 1] == 'Yes')
        self._n_something_known = np.sum(is_knowable[:, 1] != 'No')
        self._n_unknown         = np.sum(is_knowable[:, 1] == 'No')

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
            [monomials_factors[is_knowable[:, 1] == 'Yes'],
             monomials_factors[is_knowable[:, 1] == 'Semi'],
             monomials_factors[is_knowable[:, 1] == 'No']]
                                                            )

        monomials_unfactorised_reordered = np.concatenate(
            [monomials[is_knowable[:, 1] == 'Yes'],
             monomials[is_knowable[:, 1] == 'Semi'],
             monomials[is_knowable[:, 1] == 'No']]
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

    def _update_objective(self):
        for key, val in self.known_moments.items():
            if (key > 1) and (key in self._objective_as_dict.keys()):
                self._objective_as_dict[1] += self._objective_as_dict[key] * val
                del self._objective_as_dict[key]

    def _dump_to_file(self, filename):
        """
        Save the whole object to a file using `pickle`.

        Parameters
        ----------
        filename : str
            Name of the file.
        """
        import pickle
        with open(filename, 'w') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
