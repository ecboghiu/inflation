import unittest
from matplotlib.style import use
import numpy as np

from ncpol2sdpa.nc_utils import flatten
from causalinflation.quantum.general_tools import apply_source_permutation_coord_input
from causalinflation import InflationProblem, InflationSDP

class TestGeneratingMonomials(unittest.TestCase):
    bilocalDAG = {"h1": ["v1", "v2"], "h2": ["v2", "v3"]}
    inflation  = [2, 2]
    bilocality = InflationProblem(dag=bilocalDAG,
                                  settings_per_party=[1, 1, 1],
                                  outcomes_per_party=[2, 2, 2],
                                  inflation_level_per_source=inflation)
    bilocalSDP           = InflationSDP(bilocality)
    bilocalSDP_commuting = InflationSDP(bilocality, commuting=True)
    test_substitutions_scenario = InflationProblem(bilocalDAG,
                                                   settings_per_party=[1, 2, 2],
                                                   outcomes_per_party=[3, 2, 3],
                                                   inflation_level_per_source=inflation)
    # Column structure for the NPA level 2 in a tripartite scenario
    col_structure = [[],
                     [0], [1], [2],
                     [0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]
    # Monomials for the NPA level 2 in the bilocality scenario
    meas = bilocalSDP.measurements
    A_1_0_0_0 = meas[0][0][0][0]
    A_2_0_0_0 = meas[0][1][0][0]
    B_1_1_0_0 = meas[1][0][0][0]
    B_1_2_0_0 = meas[1][1][0][0]
    B_2_1_0_0 = meas[1][2][0][0]
    B_2_2_0_0 = meas[1][3][0][0]
    C_0_1_0_0 = meas[2][0][0][0]
    C_0_2_0_0 = meas[2][1][0][0]
    actual_cols = [1, A_1_0_0_0, A_2_0_0_0, B_1_1_0_0, B_1_2_0_0, B_2_1_0_0,
                   B_2_2_0_0, C_0_1_0_0, C_0_2_0_0, A_1_0_0_0*A_2_0_0_0,
                   A_1_0_0_0*B_1_1_0_0, A_1_0_0_0*B_1_2_0_0,
                   A_1_0_0_0*B_2_1_0_0, A_1_0_0_0*B_2_2_0_0,
                   A_2_0_0_0*B_1_1_0_0, A_2_0_0_0*B_1_2_0_0,
                   A_2_0_0_0*B_2_1_0_0, A_2_0_0_0*B_2_2_0_0,
                   A_1_0_0_0*C_0_1_0_0, A_1_0_0_0*C_0_2_0_0,
                   A_2_0_0_0*C_0_1_0_0, A_2_0_0_0*C_0_2_0_0,
                   B_1_1_0_0*B_1_2_0_0, B_1_1_0_0*B_2_1_0_0,
                   B_1_1_0_0*B_2_2_0_0, B_1_2_0_0*B_1_1_0_0,
                   B_1_2_0_0*B_2_1_0_0, B_1_2_0_0*B_2_2_0_0,
                   B_2_1_0_0*B_1_1_0_0, B_2_1_0_0*B_2_2_0_0,
                   B_2_2_0_0*B_1_2_0_0, B_2_2_0_0*B_2_1_0_0,
                   B_1_1_0_0*C_0_1_0_0, B_1_1_0_0*C_0_2_0_0,
                   B_1_2_0_0*C_0_1_0_0, B_1_2_0_0*C_0_2_0_0,
                   B_2_1_0_0*C_0_1_0_0, B_2_1_0_0*C_0_2_0_0,
                   B_2_2_0_0*C_0_1_0_0, B_2_2_0_0*C_0_2_0_0,
                   C_0_1_0_0*C_0_2_0_0]

    def test_generating_columns_nc(self):
        truth = 41
        columns = self.bilocalSDP.build_columns(self.col_structure,
                                                return_columns_numerical=False)
        self.assertEqual(len(columns), truth,
                         "With noncommuting variables, there are  " +
                         str(len(columns)) + " columns but " + str(truth) +
                         " were expected")

    def test_generation_from_columns(self):
        columns = self.bilocalSDP.build_columns(self.actual_cols,
                                                return_columns_numerical=False)
        self.assertEqual(columns, self.actual_cols,
                         "The direct copying of columns is failing")

    def test_generation_from_lol(self):
        columns = self.bilocalSDP.build_columns(self.col_structure,
                                                return_columns_numerical=False)
        self.assertEqual(columns, self.actual_cols,
                         "Parsing a list-of-list description of columns fails")

    def test_generation_from_str(self):
        columns = self.bilocalSDP.build_columns('npa2',
                                                return_columns_numerical=False)
        self.assertEqual(columns, self.actual_cols,
                         "Parsing the string description of columns is failing")

    def test_generate_with_identities(self):
        oneParty = InflationSDP(InflationProblem({"h": ["v"]}, [2], [2], [1]))
        _, columns = oneParty.build_columns([[], [0, 0]],
                                            return_columns_numerical=True)
        truth   = [[0],
                   [[1, 1, 0, 0], [1, 1, 1, 0]],
                   [[1, 1, 1, 0], [1, 1, 0, 0]]]
        truth = [np.array(mon) for mon in truth]
        self.assertTrue(len(columns) == len(truth), "The number of columns is incorrect.")
        areequal = np.all([np.array_equal(columns[i], truth[i]) for i in range(len(columns))])
        self.assertTrue(areequal,
                         "The column generation is not capable of handling " +
                         "monomials that reduce to the identity")

    def test_generating_columns_c(self):
        truth = 37
        columns = self.bilocalSDP_commuting.build_columns(self.col_structure,
                                                 return_columns_numerical=False)
        self.assertEqual(len(columns), truth,
                         "With commuting variables, there are  " +
                         str(len(columns)) + " columns but " + str(truth) +
                         " were expected")

    def test_nc_substitutions(self):
        settings = [1, 2, 2]
        outcomes = [3, 2, 3]
        scenario = InflationSDP(self.test_substitutions_scenario)

        meas, subs, _ = scenario._generate_parties()

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
        scenario = InflationSDP(self.test_substitutions_scenario,
                                commuting=True)
        meas, subs, _ = scenario._generate_parties()

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
        scenario = InflationSDP(InflationProblem(dag={"h": ["v"]},
                                                 outcomes_per_party=[2],
                                                 settings_per_party=[2],
                                                 inflation_level_per_source=[2]
                                                 ),
                                commuting=True)
        meas, subs, names = scenario._generate_parties()
        col_structure = [[], [0, 0]]
        flatmeas = np.array(flatten(meas))
        measnames = np.array([str(meas) for meas in flatmeas])

        # Define moment matrix columns
        _, ordered_cols_num = scenario.build_columns(col_structure,
                                                  return_columns_numerical=True)

        expected = [[0],
                    [[1, 2, 0, 0], [1, 2, 1, 0]],
                    [[1, 1, 0, 0], [1, 2, 0, 0]],
                    [[1, 1, 1, 0], [1, 2, 0, 0]],
                    [[1, 1, 0, 0], [1, 2, 1, 0]],
                    [[1, 1, 1, 0], [1, 2, 1, 0]],
                    [[1, 1, 0, 0], [1, 1, 1, 0]]]

        permuted_cols = apply_source_permutation_coord_input(ordered_cols_num,
                                                             0,
                                                             (1, 0),
                                                             False,
                                                             subs,
                                                             flatmeas,
                                                             measnames,
                                                             names)
        self.assertTrue(np.array_equal(np.array(expected[5]), permuted_cols[5]),
                         "The commuting relations of different copies are not "
                         + "being applied properly after inflation symmetries")

class TestSDPOutput(unittest.TestCase):
    def GHZ(self, v):
        dist = np.zeros((2,2,2,1,1,1))
        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    if (a == b) and (b == c):
                        dist[a,b,c,0,0,0] = v/2 + (1-v)/8
                    else:
                        dist[a,b,c,0,0,0] = (1-v)/8
        return dist

    cutInflation = InflationProblem({"lambda": ["a", "b"],
                                     "mu": ["b", "c"],
                                     "sigma": ["a", "c"]},
                                     outcomes_per_party=[2, 2, 2],
                                     settings_per_party=[1, 1, 1],
                                     inflation_level_per_source=[2, 1, 1])

    def test_CHSH(self):
        bellScenario = InflationProblem({"lambda": ["a", "b"]},
                                         outcomes_per_party=[2, 2],
                                         settings_per_party=[2, 2],
                                         inflation_level_per_source=[1])
        sdp = InflationSDP(bellScenario)
        sdp.generate_relaxation('npa1')
        self.assertEqual(len(sdp.generating_monomials), 5,
                         "The number of generating columns is not correct")
        self.assertEqual(sdp._n_known, 8,
                         "The count of knowable moments is wrong")
        self.assertEqual(sdp._n_unknown, 2,
                         "The count of unknowable moments is wrong")
        meas = sdp.measurements
        A0 = 1 - 2*meas[0][0][0][0]
        A1 = 1 - 2*meas[0][0][1][0]
        B0 = 1 - 2*meas[1][0][0][0]
        B1 = 1 - 2*meas[1][0][1][0]

        sdp.set_objective(A0*(B0+B1)+A1*(B0-B1), 'max')
        self.assertEqual(len(sdp._objective_as_dict), 7,
                         "The parsing of the objective function is failing")
        sdp.solve()
        self.assertTrue(np.isclose(sdp.objective_value, 2*np.sqrt(2)),
                        "The SDP is not recovering max(CHSH) = 2*sqrt(2)")

    def test_GHZ_NC(self):
        sdp = InflationSDP(self.cutInflation)
        sdp.generate_relaxation('local1')
        self.assertEqual(len(sdp.generating_monomials), 18,
                         "The number of generating columns is not correct")
        self.assertEqual(sdp._n_known, 8,
                         "The count of knowable moments is wrong")
        self.assertEqual(sdp._n_unknown, 13,
                         "The count of unknowable moments is wrong")

        sdp.set_distribution(self.GHZ(0.5 + 1e-4))
        self.assertEqual(sdp.known_moments[-1],
                         (0.5+1e-4)/2 + (0.5-1e-4)/8,
                         "Setting the distribution is failing")
        sdp.solve()
        self.assertEqual(sdp.status, 'infeasible',
                     "The NC SDP is not identifying incompatible distributions")
        sdp.solve(feas_as_optim=True)
        self.assertTrue(sdp.primal_objective <= 0,
                        "The NC SDP with feasibility as optimization is not " +
                        "identifying incompatible distributions")
        sdp.set_distribution(self.GHZ(0.5 - 1e-4))
        self.assertEqual(sdp.known_moments[-1],
                         (0.5-1e-4)/2 + (0.5+1e-4)/8,
                         "Re-setting the distribution is failing")
        sdp.solve()
        self.assertEqual(sdp.status, 'feasible',
                       "The NC SDP is not recognizing compatible distributions")
        sdp.solve(feas_as_optim=True)
        self.assertTrue(sdp.primal_objective >= 0,
                        "The NC SDP with feasibility as optimization is not " +
                        "recognizing compatible distributions")

    def test_GHZ_commuting(self):
        sdp = InflationSDP(self.cutInflation, commuting=True)
        sdp.generate_relaxation('local1')
        self.assertEqual(len(sdp.generating_monomials), 18,
                         "The number of generating columns is not correct")
        self.assertEqual(sdp._n_known, 8,
                         "The count of knowable moments is wrong")
        self.assertEqual(sdp._n_unknown, 11,
                         "The count of unknowable moments is wrong")

        sdp.set_distribution(self.GHZ(0.5 + 1e-2))
        sdp.solve()
        self.assertEqual(sdp.status, 'infeasible',
              "The commuting SDP is not identifying incompatible distributions")
        sdp.solve(feas_as_optim=True)
        self.assertTrue(sdp.primal_objective <= 0,
                        "The commuting SDP with feasibility as optimization " +
                        "is not identifying incompatible distributions")
        sdp.set_distribution(self.GHZ(0.5 - 1e-2))
        sdp.solve()
        self.assertEqual(sdp.status, 'feasible',
                "The commuting SDP is not recognizing compatible distributions")
        sdp.solve(feas_as_optim=True)
        self.assertTrue(sdp.primal_objective >= 0,
                        "The commuting SDP with feasibility as optimization " +
                        "is not recognizing compatible distributions")

    def test_lpi_constraints(self):
        sdp = InflationSDP(InflationProblem({"h1": ["a", "b"],
                                     "h2": ["b", "c"],
                                     "h3": ["a", "c"]},
                                     outcomes_per_party=[2, 2, 2],
                                     settings_per_party=[1, 1, 1],
                                     inflation_level_per_source=[3, 3, 3]),
                            commuting=False)
        cols = [np.array([0]),
                np.array([[1, 1, 0, 1, 0, 0]]),
                np.array([[2, 2, 1, 0, 0, 0],
                          [2, 3, 1, 0, 0, 0]]),
                np.array([[3, 0, 2, 2, 0, 0],
                          [3, 0, 3, 2, 0, 0]]),
                np.array([[1, 1, 0, 1, 0, 0],
                          [2, 2, 1, 0, 0, 0],
                          [2, 3, 1, 0, 0, 0],
                          [3, 0, 2, 2, 0, 0],
                          [3, 0, 3, 2, 0, 0]]),
        ]
        sdp.generate_relaxation(cols)
        sdp.set_distribution(self.GHZ(0.5), use_lpi_constraints=True)

        self.assertTrue(np.all(sdp.semiknown_moments[:,1] <= 1),
                    ("Semiknown moments need to be of the form " +
                    "mon_index1 = (number<=1) * mon_index2, this is failing!"))
