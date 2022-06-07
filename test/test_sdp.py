import unittest

# from sdp_utils import solveSDP
from ncpol2sdpa.nc_utils import flatten
from quantuminflation.general_tools import *
from quantuminflation.sdp_utils import *
import quantuminflation.useful_distributions as useful_distributions

from scipy.io import loadmat

""" THIS IS MISSING data/ DOES NOT WORK """

# Commented out because it takes long to test

class TestGeneratingMonomials(unittest.TestCase):
    bilocality = np.array([[1, 1, 0],
                           [0, 1, 1]])
    ins       = [1, 1, 1]
    outs      = [2, 2, 2]
    inflation = [2, 2]
    # Column structure for the NPA level 2 in a tripartite scenario
    col_structure = [[],
                     [0], [1], [2],
                     [0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]


    def test_generating_columns_nc(self):
        truth = 41
        meas, subs, names = generate_parties(self.bilocality,
                                             self.ins,
                                             self.outs,
                                             self.inflation,
                                             noncommuting=True,
                                             return_names=True)
        columns = build_columns(self.col_structure, meas, subs, names)
        self.assertEqual(len(columns), truth,
                         "With noncommuting variables, there are  " +
                         str(len(columns)) + " columns but " + str(truth) +
                         " were expected")

    def test_generate_with_identities(self):
        meas, subs, names = generate_parties(np.array([[1]]),
                                             [2],
                                             [2],
                                             [1],
                                             noncommuting=True,
                                             expectation_values=True,
                                             return_names=True)
        columns = build_columns([[0, 0]], meas, subs, names)
        truth   = [[0],
                   [[1, 1, 0, 0], [1, 1, 1, 0]],
                   [[1, 1, 1, 0], [1, 1, 0, 0]]]
        self.assertEqual(columns, truth,
                         "The column generation is not capable of handling " +
                         "monomials that reduce to the identity")

    def test_generating_columns_c(self):
        truth = 37
        meas, subs, names = generate_parties(self.bilocality,
                                             self.ins,
                                             self.outs,
                                             self.inflation,
                                             noncommuting=False,
                                             return_names=True)
        columns = build_columns(self.col_structure, meas, subs, names)
        self.assertEqual(len(columns), truth,
                         "With commuting variables, there are  " +
                         str(len(columns)) + " columns but " + str(truth) +
                         " were expected")

    def test_nc_substitutions(self):
        settings = [1, 2, 2]
        outcomes = [3, 2, 3]
        meas, subs = generate_parties(self.bilocality, settings, outcomes,
                                      self.inflation, noncommuting=True)

        true_substitutions = {}
        for party in meas:
            # Idempotency
            true_substitutions = {**true_substitutions,
                                  **{op**2: op for op in flatten(party)}}
            # Orthogonality
            for inflation in party:
                for measurement in inflation:
                    for out1 in measurement:
                        for out2 in measurement:
                            if out1 == out2:
                                true_substitutions[out1*out2] = out1
                            else:
                                true_substitutions[out1*out2] = 0
        # Commutation of different parties
        for A in flatten(meas[0]):
            for B in flatten(meas[1]):
                true_substitutions[B*A] = A*B
            for C in flatten(meas[2]):
                true_substitutions[C*A] = A*C
        for B in flatten(meas[1]):
            for C in flatten(meas[2]):
                true_substitutions[C*B] = B*C
        # Commutation of operators for nonoverlapping copies
        # Party A
        for copy1 in flatten(meas[0][0]):
            for copy2 in flatten(meas[0][1]):
                true_substitutions[copy2*copy1] = copy1*copy2
        # Party B, copies 11 and 22
        for copy1 in flatten(meas[1][0]):
            for copy2 in flatten(meas[1][3]):
                true_substitutions[copy2*copy1] = copy1*copy2
        # Party B, copies 12 and 21
        for copy1 in flatten(meas[1][1]):
            for copy2 in flatten(meas[1][2]):
                true_substitutions[copy2*copy1] = copy1*copy2
        # Party C
        for copy1 in flatten(meas[2][0]):
            for copy2 in flatten(meas[2][1]):
                true_substitutions[copy2*copy1] = copy1*copy2

        self.assertDictEqual(subs, true_substitutions)


    def test_c_substitutions(self):
        settings = [1, 2, 2]
        outcomes = [3, 2, 3]
        meas, subs = generate_parties(self.bilocality, settings, outcomes,
                                      self.inflation, noncommuting=False)

        true_substitutions = {}
        for party in meas:
            # Idempotency
            true_substitutions = {**true_substitutions,
                                  **{op**2: op for op in flatten(party)}}
            # Orthogonality
            for inflation in party:
                for measurement in inflation:
                    for out1 in measurement:
                        for out2 in measurement:
                            if out1 == out2:
                                true_substitutions[out1*out2] = out1
                            else:
                                true_substitutions[out1*out2] = 0

        self.assertDictEqual(subs, true_substitutions)

class TestInflation(unittest.TestCase):
    def test_commutations_after_symmetrization(self):
        hypergraph = np.array([[1]])
        inflation_level = [2]
        classical = False
        col_structure = [[0, 0]]

        meas, subs, names = generate_parties(hypergraph,
                                             [2],
                                             [2],
                                             inflation_level,
                                             noncommuting=True,
                                             return_names=True)
        flatmeas = np.array(flatten(meas))  # TODO remove this...
        measnames = np.array([str(meas) for meas in flatmeas])

        # Define moment matrix columns
        ordered_cols_num = build_columns(col_structure, meas, subs, names)

        expected = [[[1, 2, 0, 0]],
                    [[1, 2, 1, 0]],
                    [[1, 1, 0, 0]],
                    [[1, 1, 1, 0]],
                    [[1, 2, 0, 0], [1, 2, 1, 0]],
                    [[1, 1, 0, 0], [1, 2, 0, 0]],
                    [[1, 1, 0, 0], [1, 2, 1, 0]],
                    [[1, 2, 1, 0], [1, 2, 0, 0]],
                    [[1, 1, 1, 0], [1, 2, 0, 0]],
                    [[1, 1, 1, 0], [1, 2, 1, 0]],
                    [[1, 1, 0, 0], [1, 1, 1, 0]],
                    [[1, 1, 1, 0], [1, 1, 0, 0]]]

        permuted_cols = apply_source_permutation_coord_input(ordered_cols_num,
                                                             0,
                                                             (1, 0),
                                                             False,
                                                             subs,
                                                             flatmeas,
                                                             measnames,
                                                             names)
        self.assertEqual(expected[5], permuted_cols[5],
                         "The commuting relations of different copies are not "
                         + "being applied properly after inflation symmetries")

class TestSDPOutput(unittest.TestCase):
    '''
    def test_GHZ_known_semiknown(self):
        """
        Comparing with what I get when solving the SDP in MATLAB up to 4 decimal places.
        These lambda values are the same as in Alex's stable version 0.1 before any change to
        check hypergraphs instead of the completed connected graph.
        """

        settings_per_party = [1, 1, 1]
        outcomes_per_party = [2, 2, 2]
        hypergraph = np.array([[0, 1, 1],
                               [1, 1, 0],
                               [1, 0, 1]])  # Each line is the parties that are fed by a state
        inflation_level_per_source = [2, 2, 2]
        expectation_values = False
        col_structure = [[], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
        probability_function = useful_distributions.P_GHZ
        prob_param = 0.828  # Noise
        filename_label = 'test'
        generate_sdp_relaxation(settings_per_party, outcomes_per_party, hypergraph,
                                inflation_level_per_source, expectation_values, col_structure,
                                verbose=1.0, filename_label='', calculate_semiknowns=True, all_commuting=True)
        final_positions_matrix, known_moments, semiknown_moments, symbolic_variables_to_be_given, variable_dict \
        = helper_extract_constraints(settings_per_party, outcomes_per_party, hypergraph, inflation_level_per_source,
                                     probability_function, prob_param, filename_label='')
        #sol, lambdaval = solveSDP('inflationMATLAB_.mat', use_semiknown=True)
        #lambdaval = out[1]

        print("! Takes around 30s/14 steps per SDP. Solving with only fully-known constraints. ")
        # With seminknowns, comparing to MATLAB
        #sol, lambdaval = solveSDP('inflationMATLAB_momentmatrix_and_constraints.mat', use_semiknown=True)
        sol, lambdaval = solveSDP('inflationMATLAB_.mat', use_semiknown=True, verbosity=0)
        print("lambda=", lambdaval, "assert compares to", -0.2078, "to 4 decimal places.")
        assert abs(lambdaval - -0.2078) < 1e-4  # for reference, in MATLAB it is -0.207820

        print("! Takes around 30s/14 steps per SDP. Solving with known and semi-known constraints.")
        # Without semiknowns
        #sol, lambdaval = solveSDP('inflationMATLAB_momentmatrix_and_constraints.mat', use_semiknown=False)
        sol, lambdaval = solveSDP('inflationMATLAB_.mat', use_semiknown=False, verbosity=0)
        print("lambda=", lambdaval, "assert compares to", -0.1755, "to 4 decimal places.")
        assert abs(lambdaval - -0.1755) < 1e-4  # For refernce, in MATLAB it is -0.175523
    '''

    '''
    def test_GHZ_inf3(self):
        """
        Takes too long to solve the SDP. We will generate the SDP knowing that the result of the maximization of
        the minimum eigenvalue is deterministic, thus we will only check that the final SDP is the same as one of
        we know to be correct.
        """
        print("\nStarting to test GHZ with inflation level 3 with noise v=0.429. \n" +
              "This test calculates the SDP to solve with the constraints, and \n" +
              "checks that it is equal to a reference SDP with the correct solution.\n")
        print("Estimated time: 5-10 minutes.")
        settings_per_party = [1, 1, 1]
        outcomes_per_party = [2, 2, 2]
        hypergraph = np.array([[0, 1, 1],
                               [1, 1, 0],
                               [1, 0, 1]])  # Each line is the parties that are fed by a state
        inflation_level_per_source = [3, 3, 3]
        expectation_values = False
        col_structure = [[], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
        probability_function = useful_distributions.P_GHZ
        prob_param = 0.429  # Noise
        filename_label = 'test'
        generate_sdp_relaxation(settings_per_party, outcomes_per_party, hypergraph,
                                inflation_level_per_source, expectation_values, col_structure,
                                verbose=1.0, filename_label='', calculate_semiknowns=True, all_commuting=True)

        test_final_positions_matrix, test_known_moments, test_semiknown_moments, symbolic_variables_to_be_given, variable_dict \
        = helper_extract_constraints(settings_per_party, outcomes_per_party, hypergraph, inflation_level_per_source,
                                     probability_function, prob_param, filename_label='')


        # Import correct ones
        correct_final_positions_matrix = loadmat("test/test_data/inf3_out222_local1_P_GHZ_v0429/inflationMomentMat.mat")['G']
        correct_known_moments = loadmat("test/test_data/inf3_out222_local1_P_GHZ_v0429/inflationKnownMoments.mat")['known_moments']
        correct_semiknown_moments = loadmat("test/test_data/inf3_out222_local1_P_GHZ_v0429/inflationProptoMoments.mat")['propto']

        assert np.allclose(test_final_positions_matrix, correct_final_positions_matrix), "Not the same as the reference."
        assert np.allclose(test_known_moments, correct_known_moments), "Not the same as the reference."
        assert np.allclose(test_semiknown_moments, correct_semiknown_moments), "Not the same as the reference."
    '''
    '''
    def test_W_inf2(self):
        """
        Takes too long to solve the SDP. We will generate the SDP knowing that the result of the maximization of
        the minimum eigenvalue is deterministic, thus we will only check that the final SDP is the same as one of
        we know to be correct.
        """
        print("\nStarting to test W distrib. with inf. level 2 with noise v=0.81 and \n"
              "local level 2 truncated to operators with containing at most 4 factors. \n" +
              "This test calculates the SDP to solve with the constraints, and \n" +
              "checks that it is equal in formulation to a reference SDP which we know \n"
              "to give the correct solution.\n")
        print("Estimated time: around 5 minutes.")
        settings_per_party = [1, 1, 1]
        outcomes_per_party = [2, 2, 2]
        hypergraph = np.array([[0, 1, 1],
                               [1, 1, 0],
                               [1, 0, 1]])  # Each line is the parties that are fed by a state
        inflation_level_per_source = [2, 2, 2]
        expectation_values = False

        # local leve 2 star: only up to products of 4 terms
        col_structure = [[],
                         [0], [1], [2],
                         [0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2],
                         [0, 1, 2],
                         [0, 0, 1, 2], [0, 1, 1, 2], [0, 1, 2, 2]]
        probability_function = useful_distributions.P_W
        prob_param = 0.81  # Noise
        filename_label = 'test'
        generate_sdp_relaxation(settings_per_party, outcomes_per_party, hypergraph,
                                inflation_level_per_source, expectation_values, col_structure,
                                verbose=1.0, filename_label='', calculate_semiknowns=True, all_commuting=True)
        test_final_positions_matrix, test_known_moments, test_semiknown_moments, symbolic_variables_to_be_given, variable_dict \
        = helper_extract_constraints(settings_per_party, outcomes_per_party, hypergraph, inflation_level_per_source,
                                     probability_function, prob_param, filename_label='')

        # Import correct ones
        correct_final_positions_matrix = loadmat("test/test_data/inf2_out222_local2_star_P_W_v081/inflationMomentMat.mat")['G']
        correct_known_moments = loadmat("test/test_data/inf2_out222_local2_star_P_W_v081/inflationKnownMoments.mat")['known_moments']
        correct_semiknown_moments = loadmat("test/test_data/inf2_out222_local2_star_P_W_v081/inflationProptoMoments.mat")['propto']

        assert np.allclose(test_final_positions_matrix, correct_final_positions_matrix), "Not the same as the reference."
        assert np.allclose(test_known_moments, correct_known_moments), "Not the same as the reference."
        assert np.allclose(test_semiknown_moments, correct_semiknown_moments), "Not the same as the reference."
    '''

    '''
    def test_Mermin_inf2(self):
        """
        Takes too long to solve the SDP. We will generate the SDP knowing that the result of the maximization of
        the minimum eigenvalue is deterministic, thus we will only check that the final SDP is the same as one of
        we know to be correct.
        """
        print("\nStarting to test Mermin distrib. with inf. level 2 with noise v=0.81 and \n"
              "local level 2 truncated to operators with containing at most 4 factors. \n" +
              "This test calculates the SDP to solve with the constraints, and \n" +
              "checks that it is equal in formulation to a reference SDP which we know \n"
              "to give the correct solution.\n")
        print("Estimated time: 5-10 minutes.")
        settings_per_party = [2, 2, 2]
        outcomes_per_party = [2, 2, 2]
        hypergraph = np.array([[0, 1, 1],
                               [1, 1, 0],
                               [1, 0, 1]])  # Each line is the parties that are fed by a state
        inflation_level_per_source = [2, 2, 2]
        expectation_values = False

        # local leve 2 star: only up to products of 4 terms
        # Union of S2 and Local 1
        col_structure = [[], [0], [1], [2], [0,0], [0,1], [0,2], [1,1], [1,2], [2,2], [0,1,2]]
        probability_function = useful_distributions.P_Mermin
        prob_param = 0.51  # Noise
        filename_label = 'test'
        generate_sdp_relaxation(settings_per_party, outcomes_per_party, hypergraph,
                                inflation_level_per_source, expectation_values, col_structure,
                                verbose=1.0, filename_label='', calculate_semiknowns=True, all_commuting=True)
        test_final_positions_matrix, test_known_moments, test_semiknown_moments, symbolic_variables_to_be_given, variable_dict \
        = helper_extract_constraints(settings_per_party, outcomes_per_party, hypergraph, inflation_level_per_source,
                                     probability_function, prob_param, filename_label='')

        # Import correct ones
        correct_final_positions_matrix = loadmat("test/test_data/inf3_out222_in222_local1_S2_P_mermin_v051/inflationMomentMat.mat")['G']
        correct_known_moments = loadmat("test/test_data/inf3_out222_in222_local1_S2_P_mermin_v051/inflationKnownMoments.mat")['known_moments']
        correct_semiknown_moments = loadmat("test/test_data/inf3_out222_in222_local1_S2_P_mermin_v051/inflationProptoMoments.mat")['propto']

        assert np.allclose(test_final_positions_matrix, correct_final_positions_matrix), "Not the same as the reference."
        assert np.allclose(test_known_moments, correct_known_moments), "Not the same as the reference."
        assert np.allclose(test_semiknown_moments, correct_semiknown_moments), "Not the same as the reference."
    '''

    '''
    def test_Salman_u2_085(self):
        """
        Takes too long to solve the SDP. We will generate the SDP knowing that the result of the maximization of
        the minimum eigenvalue is deterministic, thus we will only check that the final SDP is the same as one of
        we know to be correct.
        """
        print("\nStarting to test W distrib. with inf. level 2 with noise v=0.81 and \n"
              "local level 2 truncated to operators with containing at most 4 factors. \n" +
              "This test calculates the SDP to solve with the constraints, and \n" +
              "checks that it is equal in formulation to a reference SDP which we know \n"
              "to give the correct solution.\n")
        print("Estimated time: ~30 minutes.")
        settings_per_party = [1, 1, 1]
        outcomes_per_party = [4, 4, 4]
        hypergraph = np.array([[0, 1, 1],
                               [1, 1, 0],
                               [1, 0, 1]])  # Each line is the parties that are fed by a state
        inflation_level_per_source = [2, 2, 2]
        expectation_values = False

        # local leve 2 star: only up to products of 4 terms
        # Union of S2 and Local 1
        col_structure = [[], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
        probability_function = useful_distributions.P_Salman
        prob_param = 0.85  # u2 param
        filename_label = 'GHZ_inf3'
        generate_sdp_relaxation(settings_per_party, outcomes_per_party, hypergraph,
                                inflation_level_per_source, expectation_values, col_structure,
                                verbose=1.0, filename_label='', calculate_semiknowns=True, all_commuting=True)
        test_final_positions_matrix, test_known_moments, test_semiknown_moments, symbolic_variables_to_be_given, variable_dict \
        = helper_extract_constraints(settings_per_party, outcomes_per_party, hypergraph, inflation_level_per_source,
                                     probability_function, prob_param, filename_label='')

        # Import correct ones
        correct_final_positions_matrix = \
        loadmat("test/test_data/inf2_out444_local1_P_Salman_u2_085/inflationMomentMat.mat")['G']
        correct_known_moments = \
        loadmat("test/test_data/inf2_out444_local1_P_Salman_u2_085/inflationKnownMoments.mat")['known_moments']
        correct_semiknown_moments = \
        loadmat("test/test_data/inf2_out444_local1_P_Salman_u2_085/inflationProptoMoments.mat")['propto']

        assert np.allclose(test_final_positions_matrix,
                           correct_final_positions_matrix), "Not the same as the reference."
        assert np.allclose(test_known_moments, correct_known_moments), "Not the same as the reference."
        assert np.allclose(test_semiknown_moments, correct_semiknown_moments), "Not the same as the reference."
    '''
