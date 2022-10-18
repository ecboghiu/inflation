import unittest
import numpy as np
import warnings

from causalinflation import InflationProblem, InflationSDP

trivial = InflationProblem({"h": ["v"]},
                           outcomes_per_party=[2],
                           settings_per_party=[2],
                           inflation_level_per_source=[2]
                           )


class TestMonomialGeneration(unittest.TestCase):
    bilocalDAG = {"h1": ["A", "B"], "h2": ["B", "C"]}
    bilocality = InflationProblem(dag=bilocalDAG,
                                  settings_per_party=[1, 1, 1],
                                  outcomes_per_party=[2, 2, 2],
                                  inflation_level_per_source=[2, 2])
    bilocalSDP           = InflationSDP(bilocality)
    bilocalSDP_commuting = InflationSDP(bilocality, commuting=True)
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
    cols = [1, A_1_0_0_0, A_2_0_0_0, B_1_1_0_0, B_1_2_0_0, B_2_1_0_0,
            B_2_2_0_0, C_0_1_0_0, C_0_2_0_0, A_1_0_0_0*A_2_0_0_0,
            A_1_0_0_0*B_1_1_0_0, A_1_0_0_0*B_1_2_0_0, A_1_0_0_0*B_2_1_0_0,
            A_1_0_0_0*B_2_2_0_0, A_2_0_0_0*B_1_1_0_0, A_2_0_0_0*B_1_2_0_0,
            A_2_0_0_0*B_2_1_0_0, A_2_0_0_0*B_2_2_0_0, A_1_0_0_0*C_0_1_0_0,
            A_1_0_0_0*C_0_2_0_0, A_2_0_0_0*C_0_1_0_0, A_2_0_0_0*C_0_2_0_0,
            B_1_1_0_0*B_1_2_0_0, B_1_1_0_0*B_2_1_0_0, B_1_1_0_0*B_2_2_0_0,
            B_1_2_0_0*B_1_1_0_0, B_1_2_0_0*B_2_1_0_0, B_1_2_0_0*B_2_2_0_0,
            B_2_1_0_0*B_1_1_0_0, B_2_1_0_0*B_2_2_0_0, B_2_2_0_0*B_1_2_0_0,
            B_2_2_0_0*B_2_1_0_0, B_1_1_0_0*C_0_1_0_0, B_1_1_0_0*C_0_2_0_0,
            B_1_2_0_0*C_0_1_0_0, B_1_2_0_0*C_0_2_0_0, B_2_1_0_0*C_0_1_0_0,
            B_2_1_0_0*C_0_2_0_0, B_2_2_0_0*C_0_1_0_0, B_2_2_0_0*C_0_2_0_0,
            C_0_1_0_0*C_0_2_0_0]
    actual_cols = []
    for col in cols:
        actual_cols.append(bilocalSDP._interpret_name(col))

    def test_generating_columns_c(self):
        truth = 37
        columns = self.bilocalSDP_commuting.build_columns(self.col_structure)
        self.assertEqual(len(columns), truth,
                         "With commuting variables, there are  " +
                         str(len(columns)) + " columns but " + str(truth) +
                         " were expected.")

    def test_generating_columns_nc(self):
        truth = 41
        columns = self.bilocalSDP.build_columns(self.col_structure)
        self.assertEqual(len(columns), truth,
                         "With noncommuting variables, there are  " +
                         str(len(columns)) + " columns but " + str(truth) +
                         " were expected.")

    def test_generation_from_columns(self):
        columns = self.bilocalSDP.build_columns(self.actual_cols)
        areequal = all(np.array_equal(r[0].T, np.array(r[1]).T)
                       for r in zip(columns, self.actual_cols))
        self.assertTrue(areequal, "The direct copying of columns is failing.")

    def test_generation_from_lol(self):
        columns = self.bilocalSDP.build_columns(self.col_structure)
        areequal = all(np.array_equal(r[0].T, np.array(r[1]).T)
                       for r in zip(columns, self.actual_cols))
        self.assertTrue(areequal,
                        "Parsing a list-of-list description of columns fails.")

    def test_generation_from_str(self):
        columns = self.bilocalSDP.build_columns("npa2")
        areequal = all(np.array_equal(r[0].T, np.array(r[1]).T)
                       for r in zip(columns, self.actual_cols))
        self.assertTrue(areequal,
                        "Parsing NPA levels with string description fails.")
        columns = self.bilocalSDP.build_columns("local2",
                                                max_monomial_length=2)
        diff = set(self.bilocalSDP._from_2dndarray(col) for col in columns
                   ).difference(
                       set(self.bilocalSDP._from_2dndarray(col)
                           for col in self.actual_cols))
        self.assertTrue(len(diff) == 0,
                        "Parsing local levels with string description fails.")
        columns = self.bilocalSDP.build_columns("local221",
                                                max_monomial_length=2)
        diff = set(self.bilocalSDP._from_2dndarray(col) for col in columns
                   ).difference(
                       set(self.bilocalSDP._from_2dndarray(col)
                           for col in self.actual_cols[:-1]))
        self.assertTrue(len(diff) == 0,
                        "Parsing local levels with individual string " +
                        "descriptions fails.")
        columns = self.bilocalSDP.build_columns("physical2",
                                                max_monomial_length=2)
        physical = (self.actual_cols[:22] + [self.actual_cols[24]]
                    + [self.actual_cols[26]] + self.actual_cols[32:])
        diff = set(self.bilocalSDP._from_2dndarray(col) for col in columns
                   ).difference(
                       set(self.bilocalSDP._from_2dndarray(col)
                           for col in physical))
        self.assertTrue(len(diff) == 0,
                        "Parsing physical levels with global string " +
                        "description fails.")
        columns = self.bilocalSDP.build_columns("physical",
                                                max_monomial_length=2)
        diff = set(self.bilocalSDP._from_2dndarray(col) for col in columns
                   ).difference(
                       set(self.bilocalSDP._from_2dndarray(col)
                           for col in physical))
        self.assertTrue(len(diff) == 0,
                        "Parsing physical levels without further " +
                        "description fails.")
        columns = self.bilocalSDP.build_columns("physical121",
                                                max_monomial_length=2)
        diff = set(self.bilocalSDP._from_2dndarray(col) for col in columns
                   ).difference(
                       set(self.bilocalSDP._from_2dndarray(col)
                           for col in (self.actual_cols[:9]
                                       + self.actual_cols[10:22]
                                       + [self.actual_cols[24]]
                                       + [self.actual_cols[26]]
                                       + self.actual_cols[32:-1])))
        self.assertTrue(len(diff) == 0,
                        "Parsing physical levels with individual string " +
                        "descriptions fails.")

    def test_generation_with_identities(self):
        oneParty = InflationSDP(InflationProblem({"h": ["v"]}, [2], [2], [1]))
        columns  = oneParty.build_columns([[], [0, 0]])
        truth    = [[],
                    [[1, 1, 0, 0], [1, 1, 1, 0]],
                    [[1, 1, 1, 0], [1, 1, 0, 0]]]
        truth    = [np.array(mon) for mon in truth]
        self.assertTrue(len(columns) == len(truth),
                        "Generating columns with identities is not producing "
                        + "the correct number of columns.")
        areequal = all(np.array_equiv(r[0].T, np.array(r[1]).T)
                       for r in zip(columns, truth))
        self.assertTrue(areequal,
                        "The column generation is not capable of handling " +
                        "monomials that reduce to the identity")


class TestSDPOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)

    def GHZ(self, v):
        dist = np.zeros((2, 2, 2, 1, 1, 1))
        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    if (a == b) and (b == c):
                        dist[a, b, c, 0, 0, 0] = v/2 + (1-v)/8
                    else:
                        dist[a, b, c, 0, 0, 0] = (1-v)/8
        return dist

    bellScenario = InflationProblem({"Lambda": ["A", "B"]},
                                    outcomes_per_party=[2, 2],
                                    settings_per_party=[2, 2],
                                    inflation_level_per_source=[1])

    cutInflation = InflationProblem({"lambda": ["a", "b"],
                                     "mu": ["b", "c"],
                                     "sigma": ["a", "c"]},
                                    outcomes_per_party=[2, 2, 2],
                                    settings_per_party=[1, 1, 1],
                                    inflation_level_per_source=[2, 1, 1])

    instrumental = InflationProblem({"U_AB": ["A", "B"],
                                     "A": ["B"]},
                                    outcomes_per_party=(2, 2),
                                    settings_per_party=(3, 1),
                                    inflation_level_per_source=(1,),
                                    order=("A", "B"))

    incompatible_dist = np.array([[[[0.5], [0.5], [0.0]],
                                   [[0.0], [0.0], [0.5]]],
                                  [[[0.0], [0.5], [0.0]],
                                   [[0.5], [0.0], [0.5]]]], dtype=float)

    def test_bounds(self):
        ub = 0.8
        lb = 0.2
        very_trivial = InflationProblem({"a": ["b"]}, outcomes_per_party=[2])
        sdp          = InflationSDP(very_trivial)
        sdp.generate_relaxation("npa1")
        operator = np.asarray(sdp.measurements).flatten()[0]
        sdp.set_objective(operator, "max")
        sdp.set_bounds({operator: ub}, "up")
        sdp.solve()
        self.assertTrue(np.isclose(sdp.objective_value, ub),
                        "Setting upper bounds to monomials is failing. The " +
                        f"result obtained for [max x s.t. x <= {ub}] is " +
                        f"{sdp.objective_value}.")
        sdp.set_objective(operator, "min")
        sdp.set_bounds({operator: lb}, "lo")
        sdp.solve()
        self.assertTrue(np.isclose(sdp.objective_value, lb),
                        "Setting upper bounds to monomials is failing. The " +
                        f"result obtained for [min x s.t. x >= {lb}] is " +
                        f"{sdp.objective_value}.")

    def test_equalities(self):
        prob = InflationProblem(dag={"U_AB": ["A", "B"],
                                     "U_AC": ["A", "C"],
                                     "U_AD": ["A", "D"],
                                     "C": ["D"],
                                     "A": ["B", "C", "D"]},
                                outcomes_per_party=(2, 2, 2, 2),
                                settings_per_party=(1, 1, 1, 1),
                                inflation_level_per_source=(1, 1, 1),
                                order=("A", "B", "C", "D"))
        sdp = InflationSDP(prob)
        sdp.generate_relaxation("npa2")
        equalities = sdp.moment_equalities

        self.assertEqual(len(equalities), 738,
                         "Failing to obtain the correct number of implicit " +
                         "equalities in a non-network scenario.")

        # TODO: When we add support for user-specifiable equalities, modify
        # this test to only check implicit equalities.
        self.assertTrue(all(set(eq_dict.values()) == {-1, 1}
                            for eq_dict in equalities),
                        "Some implicit equalities lack a nontrivial left-hand"
                        + "or right-hand side.")

        self.assertTrue(all(sdp.Zero.name not in eq_dict.keys()
                            for eq_dict in equalities),
                        "Some implicit equalities are wrongly assigning " +
                        "coefficients to the zero monomial.")

    def test_instrumental(self):
        sdp = InflationSDP(self.instrumental)
        sdp.generate_relaxation("local1")
        sdp.set_distribution(self.incompatible_dist)
        sdp.solve(feas_as_optim=False)
        self.assertEqual(sdp.status, "infeasible",
                         "Failing to detect the infeasibility of the " +
                         "distribution that maximally violates Bonet's " +
                         "inequalty.")
        unnormalized_dist = np.ones((2, 2, 3, 1), dtype=float)
        sdp.set_distribution(unnormalized_dist)
        sdp.solve(feas_as_optim=False)
        self.assertEqual(sdp.status, "infeasible",
                         "Failing to detect the infeasibility of an " +
                         "distribution that violates normalization.")
        compat_dist = unnormalized_dist / 4
        sdp.set_distribution(compat_dist)
        sdp.solve(feas_as_optim=False)
        self.assertEqual(sdp.status, "feasible",
                         "A feasible distribution for the instrumental " +
                         "scenario is not being recognized as such.")

    def test_instrumental_supports(self):
        sdp = InflationSDP(self.instrumental, supports_problem=True)
        sdp.generate_relaxation('local1')
        sdp.set_distribution(self.incompatible_dist)
        sdp.solve(feas_as_optim=False)
        self.assertEqual(sdp.status, "infeasible",
                         "Failing to detect the infeasibility of a support "
                         + "known to be incompatible.")
        compatible_support = np.ones((2, 2, 3, 1), dtype=float)
        sdp.set_distribution(compatible_support)
        sdp.solve(feas_as_optim=False)
        self.assertEqual(sdp.status, "feasible",
                         "A feasible support for the instrumental scenario " +
                         "is not being recognized as such.")

    def test_CHSH(self):
        sdp = InflationSDP(self.bellScenario)
        sdp.generate_relaxation("npa1")
        self.assertEqual(len(sdp.generating_monomials), 5,
                         "The number of generating columns is not correct.")
        self.assertEqual(sdp.n_knowable, 8 + 1,  # only '1' is included here.
                         "The count of knowable moments is wrong.")
        self.assertEqual(sdp.n_unknowable, 2,
                         "The count of unknowable moments is wrong.")
        meas = sdp.measurements
        A0 = 2*meas[0][0][0][0] - 1
        A1 = 2*meas[0][0][1][0] - 1
        B0 = 2*meas[1][0][0][0] - 1
        B1 = 2*meas[1][0][1][0] - 1

        sdp.set_objective(A0*(B0+B1)+A1*(B0-B1), "max")
        self.assertEqual(len(sdp.objective), 7,
                         "The parsing of the objective function is failing")
        sdp.solve()
        self.assertTrue(np.isclose(sdp.objective_value, 2*np.sqrt(2)),
                        "The SDP is not recovering max(CHSH) = 2*sqrt(2)")
        bias = 3/4
        biased_chsh = 2.62132    # Value obtained by other means (ncpol2sdpa)
        sdp.set_values({meas[0][0][0][0]: bias,    # Variable for p(a=0|x=0)
                        "A_1_1_0": bias,           # Variable for p(a=0|x=1)
                        meas[1][0][0][0]: bias,    # Variable for p(b=0|y=0)
                        "B_1_1_0": bias            # Variable for p(b=0|y=1)
                        })
        sdp.solve()
        self.assertTrue(np.isclose(sdp.objective_value, biased_chsh),
                        f"The SDP is not recovering max(CHSH) = {biased_chsh} "
                        + "when the single-party marginals are biased towards "
                        + str(bias))
        bias = 1/4
        biased_chsh = 2.55890
        sdp.set_values({meas[0][0][0][0]: bias,    # Variable for p(a=0|x=0)
                        "A_1_1_0": bias,           # Variable for p(a=0|x=1)
                        meas[1][0][0][0]: bias,    # Variable for p(b=0|y=0)
                        "B_1_1_0": bias            # Variable for p(b=0|y=1)
                        })
        sdp.solve()
        self.assertTrue(np.isclose(sdp.objective_value, biased_chsh),
                        "The SDP is not re-setting the objective correctly " +
                        "after re-setting known values.")

    def test_GHZ_commuting(self):
        sdp = InflationSDP(self.cutInflation, commuting=True)
        sdp.generate_relaxation("local1")
        self.assertEqual(len(sdp.generating_monomials), 18,
                         "The number of generating columns is not correct.")
        self.assertEqual(sdp.n_knowable, 8 + 1,  # only '1' is included here.
                         "The count of knowable moments is wrong.")
        self.assertEqual(sdp.n_unknowable, 11,
                         "The count of unknowable moments is wrong.")

        sdp.set_distribution(self.GHZ(0.5 + 1e-2))
        sdp.solve()
        self.assertEqual(sdp.status, "infeasible",
                         "The commuting SDP is not identifying incompatible " +
                         "distributions.")
        sdp.solve(feas_as_optim=True)
        self.assertTrue(sdp.primal_objective <= 0,
                        "The commuting SDP with feasibility as optimization " +
                        "is not identifying incompatible distributions.")
        sdp.set_distribution(self.GHZ(0.5 - 1e-2))
        sdp.solve()
        self.assertEqual(sdp.status, "feasible",
                         "The commuting SDP is not recognizing compatible " +
                         "distributions.")
        sdp.solve(feas_as_optim=True)
        self.assertTrue(sdp.primal_objective >= 0,
                        "The commuting SDP with feasibility as optimization " +
                        "is not recognizing compatible distributions.")

    def test_GHZ_NC(self):
        sdp = InflationSDP(self.cutInflation)
        sdp.generate_relaxation("local1")
        self.assertEqual(len(sdp.generating_monomials), 18,
                         "The number of generating columns is not correct.")
        self.assertEqual(sdp.n_knowable, 8 + 1,  # only '1' is included here.
                         "The count of knowable moments is wrong.")
        self.assertEqual(sdp.n_unknowable, 13,
                         "The count of unknowable moments is wrong.")

        sdp.set_distribution(self.GHZ(0.5 + 1e-2))
        self.assertTrue(np.isclose(sdp.known_moments[sdp.monomials[8]],
                        (0.5+1e-2)/2 + (0.5-1e-2)/8),
                        "Setting the distribution is failing.")
        sdp.solve()
        self.assertTrue(sdp.status in ["infeasible", "unknown"],
                        "The non-commuting SDP is not identifying " +
                        "incompatible distributions.")
        sdp.solve(feas_as_optim=True)
        self.assertTrue(sdp.primal_objective <= 0,
                        "The NC SDP with feasibility as optimization is not " +
                        "identifying incompatible distributions.")
        sdp.set_distribution(self.GHZ(0.5 - 1e-2))
        self.assertTrue(np.isclose(sdp.known_moments[sdp.monomials[8]],
                        (0.5-1e-2)/2 + (0.5+1e-2)/8),
                        "Re-setting the distribution is failing.")
        sdp.solve()
        self.assertEqual(sdp.status, "feasible",
                         "The non-commuting SDP is not recognizing " +
                         "compatible distributions.")
        sdp.solve(feas_as_optim=True)
        self.assertTrue(sdp.primal_objective >= 0,
                        "The NC SDP with feasibility as optimization is not " +
                        "recognizing compatible distributions.")

    def test_lpi(self):
        sdp = InflationSDP(trivial)
        [[[[A10], [A11]], [[A20], [A21]]]] = sdp.measurements
        sdp.generate_relaxation([1,
                                 A10, A11, A20, A21,
                                 A10*A11, A10*A21, A11*A20, A20*A21])
        sdp.set_distribution(np.array([[0.14873, 0.85168]]))
        # sdp.set_objective(A10*A11*A20*A21) # This produces an error
        sdp.set_objective(A11*A10*A20*A21)
        sdp.solve()
        self.assertTrue(np.isclose(sdp.objective_value, 0.0918999),
                        "Optimization of a simple SDP without LPI-like " +
                        "constraints is not obtaining the correct known value."
                        )
        sdp.set_distribution(np.array([[0.14873, 0.85168]]),
                             use_lpi_constraints=True
                             )
        sdp.solve()
        self.assertTrue(np.isclose(sdp.objective_value, 0.0640776),
                        "Optimization of a simple SDP with LPI-like " +
                        "constraints is not obtaining the correct known value."
                        )

    def test_lpi_bounds(self):
        sdp  = InflationSDP(trivial)
        cols = [[],
                [[1, 2, 0, 0],
                 [1, 2, 1, 0]],
                [[1, 1, 0, 0],
                 [1, 2, 0, 0],
                 [1, 2, 1, 0]]]
        sdp.generate_relaxation(cols)
        sdp.set_distribution(np.ones((2, 1)) / 2,
                             use_lpi_constraints=True)

        self.assertGreaterEqual(len(sdp.semiknown_moments), 1,
                                "Failing to identify semiknowns.")

        self.assertTrue(all(abs(val[0]) <= 1.
                            for val in sdp.semiknown_moments.values()),
                        "Semiknown moments need to be of the form " +
                        "mon_index1 = (number<=1) * mon_index2, this is " +
                        "failing.")


class TestSymmetries(unittest.TestCase):
    def test_commutations_after_symmetrization(self):
        scenario = InflationSDP(trivial, commuting=True)
        col_structure = [[],
                         [[1, 2, 0, 0], [1, 2, 1, 0]],
                         [[1, 1, 0, 0], [1, 2, 0, 0]],
                         [[1, 1, 1, 0], [1, 2, 0, 0]],
                         [[1, 1, 0, 0], [1, 2, 1, 0]],
                         [[1, 1, 1, 0], [1, 2, 1, 0]],
                         [[1, 1, 0, 0], [1, 1, 1, 0]]]

        scenario.generate_relaxation(col_structure)
        self.assertTrue(np.array_equal(scenario.inflation_symmetries,
                                       [[0, 6, 2, 4, 3, 5, 1]]),
                        "The commutation relations of different copies are " +
                        "not applied properly after inflation symmetries.")
