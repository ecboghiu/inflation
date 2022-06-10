import itertools
from tokenize import String
import warnings
from time import time
import os

from typing import Any, List, Dict, Union, Tuple

import sympy as sp
import numpy as np
# ncpol2sdpa >= 1.12.3 is required for quantum problems to work
from ncpol2sdpa import (SdpRelaxation, flatten)
from ncpol2sdpa.nc_utils import simplify_polynomial

from tqdm import tqdm
import itertools
import copy
import pickle

from ncpol2sdpa.nc_utils import apply_substitutions
from ncpol2sdpa import projective_measurement_constraints

from causalinflation.quantum.general_tools import (to_name, to_representative,
                                            to_numbers, mul,
                                            transform_vars_to_symb,
                                            substitute_variable_values_in_monlist,
                                            substitute_sym_with_numbers,
                                            string2prob, phys_mon_1_party_of_given_len,
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
                                            Symbolic)


from causalinflation.quantum.sdp_utils import (solveSDP_MosekFUSION,
                                        solveSDP_CVXPY, extract_from_ncpol,
                                        read_problem_from_file)

from causalinflation.quantum.fast_npa import (calculate_momentmatrix,
                                       calculate_momentmatrix_commuting,
                                       export_to_file_numba,
                                       to_canonical)

from causalinflation.quantum.writer_utils import (write_to_csv, write_to_mat,
                                           write_to_sdpa)

from causalinflation.InflationProblem import InflationProblem


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
                        Union[str, List[List[int]], List[Symbolic]] = 'npa1',
                            max_monomial_length: int = 0,
                            parallel: bool = False,
                            sandwich_positivity: bool = True,
                            use_numba: bool = True
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
        ----------
        column_specification : Union[str, List[List[int]], List[Symbolic]]
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

            * `List[Symbolic]`: we can fully specify the generating set by
            giving a list of symbolic operators built from the measurement
            operators in `self.measurements`. This list needs to have the
            identity `sympy.S.One` as the first element.

        max_monomial_length : int, optional
            Maximum number of letters in a monomial in the generating set,
            by default 0 (no limit). Example: if we choose 'local1' for 3
            parties, this gives the set {1, A, B, C, A*B, A*C, B*C, A*B*C}.
            If we set max_monomial_length=2, we remove all terms with more
            than 2 letters, and the generating set becomes:
            {1, A, B, C, A*B, A*C, B*C}.
        parallel : bool, optional
            Whether to use multiple cpus, only works with ncpol2sdpa,
            i.e., with `use_numba=False`. Note that usually Numba is faster
            than parallel ncpol2sdpa. ncpol2sdpa should only be used for
            features not present in the numba functions, such as using
            arbitrary substituion rules.
        sandwich_positivity : bool, optional
            Whether to identify monomials that are positive because of
            sandwiching. See description of `is_physical`, by default True.
        use_numba : bool, optional
            Whether to use JIT compiled functions with Numba, by default True.
            If False, the program will use ncpol2sdpa to for various steps.
            Note that usually Numba is faster than ncpol2sdpa. ncpol2sdpa
            should only be used for features not present in the numba functions,
            such as for implementing arbitrary substituion rules.
        """

        self.use_numba = use_numba
        self.use_lpi_constraints = True

        # Process the column_specification input and store the result
        #  in self.generating_monomials.
        self.build_columns(column_specification,
                           max_monomial_length=max_monomial_length)

        if self.verbose > 0:
            print("Number of columns:", len(self.generating_monomials))

        # Calculate the moment matrix without the inflation symmetries.
        problem_arr, monomials_list = self._build_momentmatrix(
                                                parallel=parallel,
                                                use_numba=use_numba)
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

        # Factorize the symmetrized monomials
        monomials_factors_names, monomials_unfactorised_reordered \
                            = self._factorize_monomials(remaining_monomials)

        unfactorized_semiknown_and_known = []
        for i in range(self._n_something_known):
            if len(monomials_factors_names[i, 1]) > 1:
                unfactorized_semiknown_and_known.append([monomials_factors_names[i]])
        unfactorized_semiknown_and_known = np.array(unfactorized_semiknown_and_known, dtype=object)

        # Combine products of unknown monomials into a single variable.
        monomials_factors_names_combined \
          = self._monomials_combine_products_of_unknown(monomials_factors_names)

        # Change the factors to representative factors
        for i in range(monomials_factors_names_combined.shape[0]):
            factors = monomials_factors_names_combined[i, 1]
            factors_rep = []
            for mon in factors:
                asnumbers = np.array(to_numbers(mon, self.names))
                asrep = to_representative(asnumbers, self.inflation_levels)
                asname = to_name(asrep,self.names)
                factors_rep.append(asname)
            #factors_rep = [ for mon in factors]
            monomials_factors_names_combined[i, 1] = factors_rep

        # Find if any of the factors cannot be achieved from the generating set,
        # and if so add them as variables.
        new_monomials_known, new_monomials_unknown \
                = self._find_new_monomials(monomials_factors_names_combined)
        if self.verbose > 0:
            print("New variables have been found not achievable " +
                                    "from the generating set of monomials:",
            "known:", new_monomials_known, "unknown:", new_monomials_unknown)

        # Update the list of monomials with the new variables.
        monomials_factors_names \
                            = monomials_factors_names_combined[:self._n_known]

        # Add new known variables
        if new_monomials_known.size > 0:
            monomials_factors_names = np.concatenate([monomials_factors_names,
                                                      new_monomials_known])

        # Add the semiknown variables, nothing different here
        monomials_factors_names = np.concatenate([
            monomials_factors_names,
            monomials_factors_names_combined[self._n_known:self._n_something_known]
                                                  ])
        # Add the new unknown variables, to which the semiknown make reference
        if new_monomials_unknown.size > 0:
            monomials_factors_names = np.concatenate([monomials_factors_names,
                                                         new_monomials_unknown])

        # Add the previous unknown variables
        monomials_factors_names = np.concatenate([
            monomials_factors_names,
            monomials_factors_names_combined[self._n_something_known:]
                                                  ])

        # Update the counts
        self._n_known += new_monomials_known.shape[0]
        self._n_something_known += new_monomials_known.shape[0]
        self._n_unknown = (monomials_factors_names_combined.shape[0] +
                           - self._n_something_known
                           + monomials_factors_names.shape[0])

        # Update orbits with the new monomials
        for row in [*new_monomials_known, *new_monomials_unknown]:
            orbits[int(row[0])] = int(row[0])

        # Reasign the integer variable names to ordered from 1 to N
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

        mon_string2int = {}
        for var_idx, [string] in monomials_unfactorised_reordered:
            mon_string2int[string] = var_idx
        for var_idx, [string] in [row.tolist()
                        for row in monomials_factors_names if len(row[1]) == 1]:
            if string not in mon_string2int:
                mon_string2int[string] = var_idx

        monomials_factors_ints = np.empty_like(monomials_factors_names)
        monomials_factors_ints[:, 0] = monomials_factors_names[:, 0]
        for i in range(monomials_factors_ints.shape[0]):
            monomials_factors_ints[i, 1] = [mon_string2int[mon]
                                    for mon in monomials_factors_names[i, 1]]

        # Now find all the positive monomials
        # TODO add also known monomials to physical_monomials
        self.physical_monomials = self._find_positive_monomials(
            monomials_factors_names, sandwich_positivity=sandwich_positivity)
        # if not find_physical_monomials:
        #     self.physical_monomials = np.array([])

        if self.verbose > 0:
            print("Number of known, semi-known and unknown variables =",
                    self._n_known, self._n_something_known-self._n_known,
                    self._n_unknown)
        if self.verbose > 0:
            print("Number of positive/physical unknown variables =",
                  len(self.physical_monomials))
            if self.verbose > 1:
                print("Positive variables:", self.physical_monomials)

        self.semiknown_reps = monomials_factors_ints[:self._n_something_known]
        self.final_monomials_list = monomials_factors_names

        self._orbits = variable_dict
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

        # Define empty arrays for conditional statement if the problem is called
        # before calling .set_distribution()
        self.known_moments = np.array([])
        self.semiknown_moments = np.array([])
        self._objective_as_dict = {1: 0.}

    def set_distribution(self, p: np.ndarray, use_lpi_constraints: bool = True) -> None:
        """Set numerically the knowable moments and semiknowable moments according
        to the probability distribution specified, p. If p is None, or the user
        doesn't pass any argument to set_distribution, then this is understood
        as a request to delete information about past distributions. If p containts
        elements that are either None or nan, then this is understood as leaving
        the corresponding variable free in the SDP approximation.
        Args:
            p (np.ndarray, optional): Multidimensional array encoding the probability vector,
            which is called as p[a,b,c,...,x,y,z,...] where a,b,c,... are outputs
            and x,y,z,... are inputs. Note: even if the inputs have cadinality 1,
            they must still be specified, and the corresponding axis dimensions are 1
            respectively.
        """
        _pdims = len(list(p.shape))
        assert _pdims % 2 == 0, "The probability distribution must have equal number of inputs and outputs"
        list(p.shape[:int(_pdims/2)]
             ) == self.InflationProblem.outcomes_per_party
        list(p.shape[int(_pdims/2):]
             ) == self.InflationProblem.settings_per_party

        self.use_lpi_constraints = use_lpi_constraints

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

        self._objective_as_dict = {}
        self.objective = 0

    def set_objective(self, objective: Symbolic,
                             direction: str ='max',
                            extraobjexpr=None) -> None:
        """Set or change the objective function of the polynomial optimization
        problem.

        Parameters
        ----------
        objective : Symbolic
            Describes the objective function.
        direction : str, optional
            Direction of the optimization (max/min), by default 'max'
        extraobjexpr : _type_, optional
            Optional parameter of a string expression of a
            linear combination of moment matrix elements to be
            included in the objective function, by default None
        """

        assert direction in ['max', 'min'], ('The direction parameter should be'
                                             + ' set to either "max" or "min"')
        if direction == 'max':
            sign = 1
            self.maximize = True
        else:
            sign = -1
            self.maximize = False

        self.objective = objective
        # self.use_lpi_constraints = False
        if extraobjexpr:
            self.extraobjexpr = extraobjexpr  # TODO process this

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
                    "values of the known moments (e.g., call self.set_distributin() " +
                    "with no input or set to nan/None ) or remove the known " +
                    "variables from the objective function.")

    def solve(self, interpreter: str='MOSEKFusion',
                    pure_feasibility_problem: bool=False,
                    solverparameters=None):
        """Call a solver on the SDP relaxation. Upon successful solution, it
        returns the primal and dual objective values along with the solution
        matrices. It also sets these values in the `sdpRelaxation` object,
        along with some status information.

        Parameters
        ----------
        interpreter : str, optional
            The solver to be called, by default 'MOSEKFusion'. It also accepts
            'CVXPY' and 'PICOS'. It is recommended to use 'MOSEKFusion'.
        pure_feasibility_problem : bool, optional
            For problems with constant objective, whether to do a pure
            feasibility problem or to relax it to an optimisation where
            we maximize the minimum eigenvalue. If such eigenvalue is negative,
            then the original problem is infeasible. By default False.
        solverparameters : _type_, optional
            Extra parmeters to be sent to the solver, by default None.

        """
        if self.momentmatrix is None:
            raise Exception("Relaxation is not generated yet. " +
                            "Call 'InflationSDP.get_relaxation(...)' first")

        semiknown_moments = self.semiknown_moments if self.use_lpi_constraints else []
        known_moments = self.known_moments if not self._objective_as_dict else [
            0, 1]

        solveSDP_arguments = {"positionsmatrix":  self.momentmatrix,
                              "objective":        self._objective_as_dict,
                              "known_vars":       known_moments,
                              "semiknown_vars":   semiknown_moments,
                              "positive_vars":    self.physical_monomials[:, 0] if self.physical_monomials.size > 0 else [],
                              "pure_feasibility_problem": pure_feasibility_problem,
                              "verbose":          self.verbose,
                              "solverparameters": solverparameters}
        if interpreter == 'CVXPY':
            sol, lambdaval = solveSDP_CVXPY(**solveSDP_arguments)
        elif interpreter == 'MOSEKFusion':
            sol, lambdaval = solveSDP_MosekFUSION(**solveSDP_arguments)
        else:
            Warning("Interpreter not found/implemented, using MOSEKFusion.")
            sol, lambdaval = solveSDP_MosekFUSION(**solveSDP_arguments)

        self.primal_objective = lambdaval
        self.solution_object = sol
        self.objective_value = lambdaval * (1 if self.maximize else -1)
        # Processed the dual certificate and stores it in
        # self.dual_certificate in various formats
        if np.array(self.semiknown_moments).size > 0:
            Warning("Beware that, because the problem contains linearized " +
                    "polynomial constraints, the certificate is not " +
                    "guaranteed to apply to other distributions")

        # if 'dual_certificate' in self.solution_object else None
        coeffs = self.solution_object['dual_certificate']
        names = self.final_monomials_list[:self._n_known]
        aux01 = np.array([[0, ['0']], [0, ['1']]], dtype=object)[:, 1]
        clean_names = np.concatenate((aux01, names[:, 1]))
        self.dual_certificate = np.array(
            [[coeffs[i], clean_names[i]] for i in range(coeffs.shape[0])], dtype=object)
        self.dual_certificate_lowerbound = 0
        self.process_certificate(as_symbols_correlators=True if all(
                         [o == 2 for o in self.InflationProblem.outcomes_per_party]) 
                                                            else False)

    def process_certificate(self,  as_symbols_probs: bool=True,
                            as_objective_function: bool=True,
                            normalize_certificate: bool=True,
                            as_symbols_operators: bool=False,
                            as_symbols_correlators: bool=False,
                            chop_tol: float=1e-8,
                            round_decimals: int=3
                            ) -> None:
        """Process the dual certificate and create symbolic instances of it.

        Currently supported to have it written in terms of probabilities
        projectors, correlators (if applicable) and as an objective function
        that can be optimized with `InflationSD`.

        Parameters
        ----------
        as_symbols_probs : bool, optional
            Give certificate as symbolic product of probabilities,
            stored in self.dual_certificate_as_symbols_operators.
            Defaults to True.
        as_objective_function : bool, optional
            Give certificate as an objective function that can be optimized
            with InflationSDP, stored in `self.dual_certificate_as_objective_function.`
            Defaults to True.
        normalize_certificate : bool, optional
            If true, eliminate all coefficients that are smaller
            than 'chop_tol' and round to the number of decimals specified
            `round_decimals`. Defaults to True.
        as_symbols_operators : bool, optional
            Give certificate as symbolic product of projectors,
            stored in self.dual_certificate_as_symbols_operators.
            Defaults to True.
        as_symbols_correlators : bool, optional
            Give certificate in correlator form, *only* if
            there are 2 outputs per party, stored in
            `self.dual_certificate_as_symbols_correlator`.
            Defaults to True.
        chop_tol : float, optional
            Coefficients in the dual certificate smaller in absolute value are
            set to zero. Defaults to 1e-8.
        round_decimals : int, optional
            Coefficients that are not set to zero are rounded to the number
            of decimals specified. Defaults to 3.

        """

        coeffs = self.dual_certificate[:, 0].astype(float)
        names = self.dual_certificate[:, 1]
        # names can still contain duplicated names, so we need to remove them

        new_dual_certificate = {tuple(name): 0 for name in names}
        for i, name in enumerate(names):
            new_dual_certificate[tuple(name)] += coeffs[i]

        coeffs = np.array(list(new_dual_certificate.values()))
        names = list(new_dual_certificate.keys())

        if normalize_certificate and not np.allclose(coeffs, 0):
            # Set to zero very small coefficients
            coeffs[np.abs(coeffs) < chop_tol] = 0
            # Take the smallest one and make it 1
            coeffs /= np.abs(coeffs[np.abs(coeffs) > chop_tol]).min()
            # Round to 5 decimal places
            coeffs = np.round(coeffs, decimals=round_decimals)

        self.dual_certificate_as_symbols_operators = None
        if as_symbols_operators:
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
                            # Omitting the inflation indices!
                            label = letter[0] + '_' + letter[-3:]
                            op = sp.Symbol(label, commutative=False)
                            monomial *= op
                monomial *= coeffs[i]
                polynomial += monomial
            self.dual_certificate_as_symbols_operators = polynomial

        self.dual_certificate_as_objective_function = None
        if as_objective_function:
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
            self.dual_certificate_as_objective_function = polynomial

        self.dual_certificate_as_symbols_probs = None
        if as_symbols_probs:
            polynomial = 0
            for i, row in enumerate(names):
                asprobs = [string2prob(
                    term, self.InflationProblem.nr_parties) for term in row]
                monomial = sp.S.One
                for prob in asprobs:
                    monomial *= prob
                monomial *= coeffs[i]
                polynomial += monomial
            self.dual_certificate_as_symbols_probs = polynomial

        self.dual_certificate_as_symbols_correlators = None
        if as_symbols_correlators:
            if not all([o == 2 for o in self.InflationProblem.outcomes_per_party]):
                raise Exception("Correlator certificates are only available " +
                                "for 2-output problems")

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
                                    base = sp.Symbol(auxname, commutative=True)
                                    expected_value = base ** exp
                                else:
                                    auxname = '\langle ' + ' '.join(str(var).split('*')) + ' \\rangle'
                                    auxname = '<'+ ''.join(str(var).split('*')) + '>'
                                    expected_value = sp.Symbol(auxname, commutative=True)
                            suma += coeff*expected_value
                        poly1 *= suma
                    else:
                        if row[0] == '1':
                            poly1 = sp.S.One
                        elif row[0] == '0':
                            poly1 = sp.S.Zero
                polynomial += coeffs[i]*poly1
            self.dual_certificate_as_symbols_correlators = sp.expand(polynomial)

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
            # Moment matrix in readable form, objective in the first cell
        elif extension == 'mat':
            write_to_mat(self, filename)

    def build_columns(self, column_specification, max_monomial_length=0) -> None:
        """Process the input for the columns of the SDP relaxation.

        Parameters
        ----------
        column_specification : Union[str, List[List[int]], List[Symbolic]]
            See description in the self.generate_relaxation()` method.
        max_monomial_length : int, optional
            See description in the self.generate_relaxation()` method,
            by default 0 (unbounded).
        """
        if type(column_specification) == list:
            strtype = str(type(column_specification))
            if len(strtype) > 8 and strtype[:8] == "<class 'sympy":
                self.generating_monomials = []
                for col in column_specification:
                    if col == sp.S.One:
                        self.generating_monomials += [[]]
                    else:
                        self.generating_monomials += [
                            to_numbers(str(col), self.names)]

            # == 1+2+self.InflationProblem.nr_sources:
            elif len(np.array(column_specification[1]).shape) > 1:
                self.generating_monomials = column_specification


            else:
                if max_monomial_length > 0:
                    to_remove = []
                    for idx, col_spec in enumerate(column_specification):
                        if len(col_spec) > max_monomial_length:
                            to_remove.append(idx)
                    column_specification_new = column_specification.copy()
                    for idx in sorted(to_remove, reverse=True):
                        del column_specification_new[idx]
                    col_specs = column_specification_new
                else:
                    col_specs = column_specification

                self._build_cols_from_col_specs(col_specs)


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
                self._build_cols_from_col_specs(col_specs)


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

                self._build_cols_from_col_specs(col_specs)


            elif 'physical' in column_specification.lower():
                try:
                    inf_level = int(column_specification[8])
                    length = len(column_specification[8:])
                    message = ("Physical monomial generating set party number" +
                               "specification must have length equal to " +
                               "number of parties. E.g.: For 3 parties, " +
                               "'physical322'.")
                    assert length <= self.nr_parties and length > 0, message
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

                self.generating_monomials = physical_monomials

            else:
                raise Exception('I have not understood the format of the '
                                + 'column specification')
        else:
            raise Exception('I have not understood the format of the '
                            + 'column specification')

        self.generating_monomials_sym = \
                from_coord_to_sym(self.generating_monomials,
                                    self.names,
                                    self.InflationProblem.nr_sources,
                                    self.measurements)

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
        and "local level 1" is build from
        [[], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]


        Parameters
        ----------
        col_specs : Union[str, List[List[int]], List[Symbolic]]
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
                warnings.warn("You have not input substitution rules to the " +
                              "generation of columns, but it is OK because you " +
                              "are using local level 1")
            else:
                raise Exception("You must input substitution rules for columns " +
                                "to be generated properly")

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
                    for slicee in itertools.product(*meas_ops):
                        monomial = apply_substitutions(
                            np.prod(slicee), self.substitutions)
                        if monomial not in symbols:
                            symbols.append(monomial)
                            if monomial == 1:
                                coords = [0]
                            else:
                                coords = []
                                for factor in monomial.as_coeff_mul()[1]:
                                    coords.append(*to_numbers(factor, self.names))
                            res.append(coords)

            self.generating_monomials = sorted(res, key=len)

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
        settings = self.InflationProblem.settings_per_party
        outcomes = self.InflationProblem.outcomes_per_party
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

    def _build_momentmatrix(self, parallel: bool=False,
                                  use_numba: bool=True
                            ) -> None:
        """Generate the moment matrix or load it from file if it is already calculated.

        Parameters
        ----------
        parallel : bool, optional
            Specifies whether to use parallelization or not.
            Only works with ncpol2sdpa. Defaults to False.
        use_numba : bool, optional
            Whether to use JIT functions through numba to calculate
            the moment matrix. Defaults to True.
        """

        if use_numba:
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

            if self.verbose >= 2:
                export_to_file_numba(problem_arr, monomials_list)
            # TODO change from dense to sparse !! Else useless, but this requires adapting code
            problem_arr = problem_arr.todense()
        else:
            time0 = time()
            sdp = SdpRelaxation(flatten(self.measurements),
                                verbose=self.verbose, parallel=parallel)
            sdp.get_relaxation(level=-1,
                               extramonomials=self.generating_monomials_sym,
                               substitutions=self.substitutions)

            if self.verbose >= 1:
                print("SDP relaxation was generated in " +
                      str(time() - time0) + " seconds.")
                if self.verbose >= 2:
                    print("Saving as '" + filename_momentmatrix +
                          "' and '" + filename_monomials + "'")
                print("")
            sdp.write_to_file('debug_momentmatrix.dat-s')
            sdp.save_monomial_index('debug_monomials.txt')

            problem_arr, _, monomials_list = extract_from_ncpol(sdp,
                                                                self.verbose)
        # monomials_list[:, 0] = monomials_list[:, 0].astype(int)  # TODO: make it work with int
        return problem_arr, monomials_list

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


        inflevel = self.InflationProblem.inflation_level_per_source
        n_sources = len(inflevel)
        # Start with the identity permutation
        inflation_symmetries = [list(range(len(self.generating_monomials)))]

        # TODO do this function without relying on symbolic substitutions!!
        flatmeas = np.array(flatten(self.measurements))
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
            total_perm = find_permutation(list_permuted, list_original)
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
                    warnings.warn("Your generating set might not have enough" +
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
        self._n_uknown = np.sum(monomials_factors_knowable[:, 1] == 'No')

        # Reorder according to know, semiknown and unknown.
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

    def _monomials_combine_products_of_unknown(self,
                                              monomials_factors_names: np.ndarray
                                              ) -> np.ndarray:
        """In the factorised monomials, combine the products of factors with
        unknown values

        TODO: Check by splitting into factors and recombining, we are
        generating new monomials.

        Parameters
        ----------
        monomials_factors_names : np.ndarray
            Monomials factorised as List[Tuple[int, List[ArrayMonomial]]].

        Returns
        -------
        np.ndarray
            Same format as input, but with the products of unknown factors
            combined.
        """
        # monomials_factors_names is reordered according to known, semiknown and unknown

        monomials_factors_names_combined = monomials_factors_names.copy()

        for idx, line in enumerate(monomials_factors_names_combined[self._n_known:]):
            var = line[0]
            factors = np.array(line[1])

            if len(factors) > 1:
                # TODO doing this twice, maybe reuse the previous calc?? more efficient
                where_unknown = np.array(
                    [not is_knowable(to_numbers(f, self.names), self.hypergraph) for f in factors])
                factors_unknown = factors[where_unknown]
                factors_known = factors[np.invert(where_unknown)]

                if self.use_numba:
                    joined_unknowns_name = '*'.join(factors_unknown)
                    joined_unknowns_name_canonical = to_name(to_canonical(
                        np.array(to_numbers(joined_unknowns_name, self.names))), self.names)
                    new_line = [var, factors_known.tolist(
                    ) + [joined_unknowns_name_canonical]]
                    monomials_factors_names_combined[idx +
                                                     self._n_known] = new_line
                else:
                    flatmeas = np.array(flatten(self.measurements))
                    measnames = np.array([str(meas) for meas in flatmeas])

                    unknown_components = flatten(
                        [part.split('*') for part in factors_unknown])
                    unknown_operator = mul(
                        [flatmeas[measnames == op] for op in unknown_components])[0]
                    unknown_operator = apply_substitutions(
                        unknown_operator, self.substitutions)

                    # It seems like ncpol2sdpa does not use as canonical form a
                    # lexicographic order, but the first representation appearing,
                    # which can be the adjoint of the lexicographic form. The following
                    # loop tries to fix this
                    # # # if sum(monomials_list[:, 1] == str(unknown_operator)) == 0:
                    # # #     unknown_operator = apply_substitutions(unknown_operator.adjoint(),
                    # # #                                         self.substitutions)
                    #unknown_var = int(monomials_list[monomials_list[:, 1] == str(unknown_operator)][0][0])
                    new_line = [var, factors_known.tolist() +
                                [str(unknown_operator)]]
                    monomials_factors_names_combined[idx +
                                                     self._n_known] = new_line

        return monomials_factors_names_combined

        # return monomials_factors_names_reordered, monomials_factors_knowable, semiknown_vars, new_monomials_list, orbits
    def _find_new_monomials(self, monomials_factors_names: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """As input we get the list of monomials factorised. The monomials with
        just a single factor are monomials that can be found from products
        of operators from the generating set. If there are two or more factors,
        some of the factors might not be reachable from the generating set.
        We then add these as new monomials. We also label them as known or
        unknown.

        Parameters
        ----------
        monomials_factors_names : np.ndarray
            Input monomials.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            New monomials found in the factorsand, first all known and then
            all unknown.
        """

        last_index = np.max(monomials_factors_names[:, 0]) + 1

        monomials_1factor = monomials_factors_names[[
            len(mon) == 1 for mon in monomials_factors_names[:, 1]]][:, 1]
        monomials_1factor = [i[0] for i in monomials_1factor]  # formatting
        # set(monomials_1factor)  # They should be unique, so no need for the set function
        old_monomials = set(monomials_1factor)

        monomials_morethan2factors = monomials_factors_names[[
            len(mon) > 1 for mon in monomials_factors_names[:, 1]]][:, 1]
        monomials_morethan2factors_temp = []
        for mon in monomials_morethan2factors:
            monomials_morethan2factors_temp += mon
        new_monomials_candidates = set(monomials_morethan2factors_temp)

        new_monomials_known = []
        new_monomials_unknown = []
        for mon_ in new_monomials_candidates:
            mon = to_name(to_representative(
                np.array(to_numbers(mon_, self.names)), self.inflation_levels), self.names)
            if mon not in old_monomials:
                knowable = is_knowable(to_numbers(
                    mon, self.names), self.hypergraph)
                if knowable:
                    new_monomials_known.append([last_index, [mon]])
                else:
                    new_monomials_unknown.append([last_index, [mon]])
                old_monomials.add(mon)
                last_index += 1

        new_monomials_known = np.array(new_monomials_known, dtype=object)
        new_monomials_unknown = np.array(new_monomials_unknown, dtype=object)

        return new_monomials_known, new_monomials_unknown

    def _find_positive_monomials(self, monomials_factors_names: np.ndarray,
                                 sandwich_positivity=False):
        ispositive = np.empty_like(monomials_factors_names)
        ispositive[:, 0] = monomials_factors_names[:, 0]
        ispositive[:, 1] = False
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
        return monomials_factors_names[ispositive[:, 1].astype(bool)]

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
