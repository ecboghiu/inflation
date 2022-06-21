import copy
import itertools
import numpy as np
import pickle
import sympy as sp

from causalinflation import InflationProblem
from causalinflation.quantum.general_tools import (to_name, to_representative,
                                            to_numbers, mul,
                                            transform_vars_to_symb,
                                            substitute_variable_values_in_monlist,
                                            substitute_sym_with_numbers,
                                            string2prob,
                                            phys_mon_1_party_of_given_len,
                                            is_knowable, is_physical,
                                            label_knowable_and_unknowable,
                                            monomialset_name2num,
                                            monomialset_num2name,
                                            factorize_monomials,
                                            find_permutation,
                                            apply_source_permutation_coord_input,
                                            from_numbers_to_flat_tuples,
                                            generate_commuting_measurements,
                                            generate_noncommuting_measurements,
                                            from_coord_to_sym,
                                            get_variables_the_user_can_specify,
                                            clean_coefficients)
from causalinflation.quantum.fast_npa import (calculate_momentmatrix,
                                              calculate_momentmatrix_commuting,
                                              to_canonical)
from causalinflation.quantum.sdp_utils import solveSDP_MosekFUSION
from causalinflation.quantum.writer_utils import (write_to_csv, write_to_mat,
                                                  write_to_sdpa)
# ncpol2sdpa >= 1.12.3 is required for quantum problems to work
from ncpol2sdpa import flatten, projective_measurement_constraints
from ncpol2sdpa.nc_utils import apply_substitutions, simplify_polynomial
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
                            max_monomial_length=0,
                            return_columns_numerical=True)

        if self.verbose > 0:
            print("Number of columns:", len(self.generating_monomials))

        # Calculate the moment matrix without the inflation symmetries.
        problem_arr, monomials_list, mon_string2int = self._build_momentmatrix()
        self._monomials_list_all = monomials_list

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
                                                              monomials_list,
                                                              inflation_symmetries)
        # Associate the names of all copies to the same variable
        for key, val in mon_string2int.items():
            mon_string2int[key] = orbits[val]
        # Factorize the symmetrized monomials
        monomials_factors_names, monomials_unfactorised_reordered \
                            = self._factorize_monomials(remaining_monomials)

        unfactorized_semiknown_and_known = []
        for i in range(self._n_something_known):
            if len(monomials_factors_names[i, 1]) > 1:
                unfactorized_semiknown_and_known.append([monomials_factors_names[i]])
        unfactorized_semiknown_and_known = np.array(unfactorized_semiknown_and_known, dtype=object)

        # Reassign the integer variable names to ordered from 1 to N
        variable_dict = {**{0: 0, 1: 1},
                         **dict(zip(monomials_factors_names[:, 0],
                              range(2, monomials_factors_names.shape[0] + 2)))}
        monomials_factors_names2 = monomials_factors_names.copy()
        for i in range(monomials_factors_names.shape[0]):
            monomials_factors_names2[i, 0] = \
                                 variable_dict[monomials_factors_names[i, 0]]
        monomials_factors_names[:, 0] = \
                            np.arange(2, monomials_factors_names.shape[0] + 2)

        for i in range(monomials_unfactorised_reordered.shape[0]):
            monomials_unfactorised_reordered[i, 0] = \
                         variable_dict[monomials_unfactorised_reordered[i, 0]]

        monomials_factors_ints = np.empty_like(monomials_factors_names)
        monomials_factors_ints[:, 0] = monomials_factors_names[:, 0]
        for i in range(monomials_factors_ints.shape[0]):
            monomials_factors_ints[i, 1] = [mon_string2int[mon]
                                    for mon in monomials_factors_names[i, 1]]

        # Now find all the positive monomials
        if self.commuting:
            self.physical_monomials = monomials_factors_names[:,0]
        else:
            self.physical_monomials = self._find_positive_monomials(
                monomials_factors_names, sandwich_positivity=True)

        if self.verbose > 0:
            print("Number of known, semi-known and unknown variables =",
                    self._n_known, self._n_something_known-self._n_known,
                    self._n_unknown)
        if self.verbose > 0:
            print("Number of positive unknown variables =",
                  len(self.physical_monomials) - self._n_known)
            if self.verbose > 1:

        self.semiknown_reps = monomials_factors_ints[:self._n_something_known]
        self.final_monomials_list = monomials_factors_names

        self._orbits   = variable_dict
        self._var2repr = orbits
        self._mon2indx = mon_string2int

        # Change moment matrix to the new integer variables
        positions_matrix = symmetric_arr[:, :, 0].astype(int)
        self.momentmatrix = positions_matrix.copy()
        for i, row in enumerate(tqdm(self.momentmatrix,
                                     disable=not self.verbose,
                                     desc="Reassigning moment matrix indices")):
            for j, col in enumerate(row):
                self.momentmatrix[i, j] = variable_dict[col]
                print("Positive variables:",
                      [self.final_monomials_list[phys-2]
                                           for phys in self.physical_monomials])

        # Define empty arrays for conditional statement if the problem is called
        # before calling .set_distribution()
        self.known_moments      = np.array([0, 1])
        self.semiknown_moments  = np.array([])
        self._objective_as_dict = {1: 0.}

    def set_distribution(self,
                         p: np.ndarray,
                         use_lpi_constraints: bool = False) -> None:
        """Set numerically the knowable moments and semiknowable moments according
        to the probability distribution specified, p. If p is None, or the user
        doesn't pass any argument to set_distribution, then this is understood
        as a request to delete information about past distributions. If p containts
        elements that are either None or nan, then this is understood as leaving
        the corresponding variable free in the SDP approximation.
        Args:
            p (np.ndarray): Multidimensional array encoding the probability
            vector, which is called as p[a,b,c,...,x,y,z,...] where a,b,c,...
            are outputs and x,y,z,... are inputs. Note: even if the inputs have
            cardinality 1, they must still be specified, and the corresponding
            axis dimensions are 1.

            use_lpi_constraints (bool): Specification whether linearized
            polynomial constraints (see, e.g., Eq. (D6) in arXiv:2203.16543)
            will be imposed or not.
        """
        _pdims = len(list(p.shape))
        assert _pdims % 2 == 0, "The probability distribution must have equal number of inputs and outputs"
        list(p.shape[:int(_pdims/2)]
             ) == self.InflationProblem.outcomes_per_party
        list(p.shape[int(_pdims/2):]
             ) == self.InflationProblem.settings_per_party

        self.use_lpi_constraints = use_lpi_constraints

        if (len(self._objective_as_dict) > 1) and self.use_lpi_constraints:
            warn("You have an objective function set. Be aware that imposing " +
                 "linearized polynomial constraints will constrain the " +
                 "optimization to distributions with fixed marginals.")

        if self.use_lpi_constraints:
            stop_counting = self._n_something_known
        else:
            stop_counting = self._n_known

        # Extract variables whose value we can get from the probability distribution in a symbolic form
        variables_to_be_given = get_variables_the_user_can_specify(
            self.semiknown_reps[:self._n_known], self.final_monomials_list[:self._n_known])
        self.symbolic_variables_to_be_given = transform_vars_to_symb(copy.deepcopy(variables_to_be_given),  # TODO change this name
                                                                     max_nr_of_parties=self.InflationProblem.nr_parties)
        if self.verbose > 1:
            print("Simplest known variables: =",
                  self.symbolic_variables_to_be_given)

        # Substitute the list of known variables with symbolic values with numerical values
        variables_values = substitute_sym_with_numbers(copy.deepcopy(self.symbolic_variables_to_be_given),
                                                       self.InflationProblem.settings_per_party,
                                                       self.InflationProblem.outcomes_per_party,
                                                       p)

        assert (np.array(variables_values, dtype=object)[:, 0].astype(int).tolist()
                == np.array(variables_to_be_given, dtype=object)[:, 0].astype(int).tolist())
        if self.verbose > 1:
            print("Variables' numerical values given p =", variables_values)
        final_monomials_list_numerical = substitute_variable_values_in_monlist(variables_values, self.semiknown_reps,
                                                                               self.final_monomials_list, stop_counting)

        self.known_moments = np.array(
            [0, 1] + [mul(factors) for _, factors in final_monomials_list_numerical[:self._n_known]])


        # The indices for the variables in the semiknowns also need shifting
        self.semiknown_moments = []
        if self.use_lpi_constraints:
            self.semiknown_moments = np.array([[var, mul(val[:-1]), val[-1]]
                                               for var, val in final_monomials_list_numerical[self._n_known:self._n_something_known]])


    def set_objective(self, objective: sp.core.symbol.Symbol,
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
            # Write objective as dictionary
            string2int_dict = {**{'0': '0', '1': '1'},
                               **dict(self._monomials_list_all[:, ::-1])}
            objective_as_dict = objective.as_coefficients_dict()
            # Express objective in terms of representatives
            symmetrized_objective = {1: 0.}
            for monomial, coeff in objective_as_dict.items():
                monomial_variable = int(string2int_dict[str(monomial)])
                # This has to be fixed carefully. We want one single
                # dictionary for bringing any variable to its representative.
                # Why do we need two steps?
                reprr = self._orbits[self._var2repr[monomial_variable]]
                if reprr in symmetrized_objective.keys():
                    symmetrized_objective[reprr] += sign * coeff
                else:
                    symmetrized_objective[reprr] = sign * coeff
            self._objective_as_dict = symmetrized_objective
        else:
            self._objective_as_dict = {1: sign * float(objective)}

        # If there is a conflict between fixed known moments
        # and variables in the objective
        vars_in_objective = self._objective_as_dict.keys()
        vars_known = [i for i in range(
            2, len(self.known_moments)) if self.known_moments[i] != np.nan]
        for var in vars_known:
            if var in vars_in_objective:
                raise Exception(
                    "You have variables in the objective that are also known " +
                    "moments fixed by a distribution. Either erase the fixed " +
                    "values of the known moments (e.g., call self.set_distribution() " +
                    "with no input or set to nan/None ) or remove the known " +
                    "variables from the objective function.")

    def solve(self, interpreter: str='MOSEKFusion',
                    feas_as_optim: bool=False,
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
        solverparameters : _type_, optional
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
        if np.array(self.semiknown_moments).size > 0:
            warn("Beware that, because the problem contains linearized " +
                 "polynomial constraints, the certificate is not guaranteed " +
                 "to apply to other distributions")

        semiknown_moments = self.semiknown_moments if self.use_lpi_constraints else []
        known_moments     = self.known_moments

        solveSDP_arguments = {"positionsmatrix":  self.momentmatrix,
                              "objective":        self._objective_as_dict,
                              "known_vars":       known_moments,
                              "semiknown_vars":   semiknown_moments,
                              "positive_vars":    self.physical_monomials,
                              "feas_as_optim":    feas_as_optim,
                              "verbose":          self.verbose,
                              "solverparameters": solverparameters}

        sol, lambdaval = solveSDP_MosekFUSION(**solveSDP_arguments)

        # Process the solution
        self.primal_objective = lambdaval
        self.solution_object  = sol
        self.objective_value  = lambdaval * (1 if self.maximize else -1)
        # Process the dual certificate in a generic form
        coeffs      = self.solution_object['dual_certificate']
        names       = self.final_monomials_list[:self._n_known]
        aux01       = np.array([[0, ['0']], [0, ['1']]], dtype=object)[:, 1]
        clean_names = np.concatenate((aux01, names[:, 1]))
        self.dual_certificate = np.array(list(zip(coeffs, clean_names)),
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
        # C: why did I write this??
        # names can still contain duplicated names, so we need to remove them
        # new_dual_certificate = {tuple(name): 0 for name in names}
        # for i, name in enumerate(names):
        #     new_dual_certificate[tuple(name)] += coeffs[i]
        # coeffs = np.array(list(new_dual_certificate.values()))
        # names = list(new_dual_certificate.keys())

        if clean and not np.allclose(coeffs, 0):
            coeffs = clean_coefficients(coeffs, chop_tol, round_decimals)

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
        if clean and not np.allclose(coeffs, 0):
            coeffs = clean_coefficients(coeffs, chop_tol, round_decimals)
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
        if clean and not np.allclose(coeffs, 0):
            coeffs = clean_coefficients(coeffs, chop_tol, round_decimals)

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
                      return_columns_numerical: bool = True) -> None:
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
            if type(column_specification[0]) == list:
                columns = self._build_cols_from_col_specs(column_specification)
            else:
                columns = []
                for col in column_specification:
                    if col == sp.S.One or col == 1:
                        columns += [[]]
                    else:
                        columns += [to_numbers(str(col), self.names)]
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
        """his builds the generating set for the moment matrix taking as input
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
            return sorted(res, key=len)

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
            list_permuted = from_numbers_to_flat_tuples(permuted_cols_ind)
            total_perm    = find_permutation(list_permuted, list_original)
            inflation_symmetries.append(total_perm)

        return inflation_symmetries

    def _apply_inflation_symmetries(self, momentmatrix: np.ndarray,
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

    def _factorize_monomials(self, remaining_monomials: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """Splits the monomials into factors according to the supports of the
        operators.

        Parameters
        ----------
        remaining_monomials : np.ndarray
            List of unfactorised monomials.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The factorised monomials reordered according to know, semiknown
            and unknown moments and also the corresponding unfactorised
            monomials, also reordered.
        """
        # TODO see why I cannot just plug in remaining_monomials instead of monomials_list??
        monomials_factors = factorize_monomials(monomialset_name2num(
            remaining_monomials, self.names), verbose=self.verbose)
        monomials_factors_names = monomialset_num2name(
            monomials_factors, self.names)
        monomials_factors_knowable = label_knowable_and_unknowable(
            monomials_factors, self.InflationProblem.hypergraph)

        # Some counting
        self._n_known = np.sum(monomials_factors_knowable[:, 1] == 'Yes')
        self._n_something_known = np.sum(
            monomials_factors_knowable[:, 1] != 'No')
        self._n_unknown = np.sum(monomials_factors_knowable[:, 1] == 'No')

        # Reorder according to known, semiknown and unknown.
        monomials_factors_names_reordered = np.concatenate(
            [monomials_factors_names[monomials_factors_knowable[:, 1] == 'Yes'],
             monomials_factors_names[monomials_factors_knowable[:, 1] == 'Semi'],
             monomials_factors_names[monomials_factors_knowable[:, 1] == 'No']]
                                                            )

        monomials_unfactorised_reordered = np.concatenate(
            [remaining_monomials[monomials_factors_knowable[:, 1] == 'Yes'],
             remaining_monomials[monomials_factors_knowable[:, 1] == 'Semi'],
             remaining_monomials[monomials_factors_knowable[:, 1] == 'No']]
                                                            )
        monomials_unfactorised_reordered = monomials_unfactorised_reordered.astype(object)
        monomials_unfactorised_reordered[:, 0] = monomials_unfactorised_reordered[:, 0].astype(int)
        monomials_unfactorised_reordered[:, 1] = [[mon] for mon in monomials_unfactorised_reordered[:, 1]]

        return monomials_factors_names_reordered, monomials_unfactorised_reordered

    def _find_positive_monomials(self, monomials_factors_names: np.ndarray,
                                       sandwich_positivity=True):
        ispositive = np.empty_like(monomials_factors_names)
        ispositive[:, 0] = monomials_factors_names[:, 0]
        ispositive[:, 1] = False
        ispositive[:self._n_known, 1] = True    # Knowable moments are physical
        for i, row in enumerate(monomials_factors_names[self._n_known:]):
            factors = row[1]
            factor_is_positive = []
            for mon in factors:
                asnumbers = np.array(to_numbers(mon, self.names))
                isphysical = is_physical(asnumbers,
                                         sandwich_positivity=sandwich_positivity)
                factor_is_positive.append(isphysical)
            if all(factor_is_positive):
                ispositive[i+self._n_known, 1] = True

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
