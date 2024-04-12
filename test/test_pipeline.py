import unittest
import numpy as np
import warnings
from itertools import product

from inflation import InflationProblem, InflationSDP, InflationLP

bilocalDAG = {"h1": ["A", "B"], "h2": ["B", "C"]}
bilocality = InflationProblem(dag=bilocalDAG,
                              settings_per_party=[1, 1, 1],
                              outcomes_per_party=[2, 2, 2],
                              inflation_level_per_source=[2, 2])
bilocality_c = InflationProblem(dag=bilocalDAG,
                                settings_per_party=[1, 1, 1],
                                outcomes_per_party=[2, 2, 2],
                                inflation_level_per_source=[2, 2],
                                classical_sources="all")
bilocalSDP = InflationSDP(bilocality)

trivial = InflationProblem({"h": ["v"]},
                           outcomes_per_party=[2],
                           settings_per_party=[2],
                           inflation_level_per_source=[2]
                           )
trivial_c = InflationProblem({"h": ["v"]},
                           outcomes_per_party=[2],
                           settings_per_party=[2],
                           inflation_level_per_source=[2],
                           classical_sources="all"
                           )


class TestMonomialGeneration(unittest.TestCase):
    bilocalSDP_commuting = InflationSDP(bilocality_c)
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
        actual_cols.append(bilocalSDP.mon_to_lexrepr(bilocalSDP._interpret_name(col)))
    actual_cols_2d = [bilocalSDP._lexorder[lexmon] for lexmon in actual_cols]

    def test_generating_columns_c(self):
        truth = 37
        columns = self.bilocalSDP_commuting.build_columns(self.col_structure)
        self.assertEqual(len(columns), truth,
                         "With commuting variables, there are  " +
                         str(len(columns)) + " columns but " + str(truth) +
                         " were expected.")

    def test_generating_columns_nc(self):
        truth = 41
        columns = bilocalSDP.build_columns(self.col_structure)
        self.assertEqual(len(columns), truth,
                         "With noncommuting variables, there are  " +
                         str(len(columns)) + " columns but " + str(truth) +
                         " were expected.")

    def test_generation_from_columns(self):
        columns = bilocalSDP.build_columns(self.actual_cols_2d)
        areequal = all(np.array_equal(r[0].T, np.array(r[1]).T)
                       for r in zip(columns, self.actual_cols))
        self.assertTrue(areequal, "The direct copying of columns is failing.")

    def test_generation_from_lol(self):
        columns = bilocalSDP.build_columns(self.col_structure)
        areequal = all(np.array_equal(r[0].T, np.array(r[1]).T)
                       for r in zip(columns, self.actual_cols))
        self.assertTrue(areequal,
                        "Parsing a list-of-list description of columns fails.")

    def test_generation_from_str(self):
        columns = bilocalSDP.build_columns("npa2")
        areequal = all(np.array_equal(r[0].T, np.array(r[1]).T)
                       for r in zip(columns, self.actual_cols))
        self.assertTrue(areequal,
                        "Parsing NPA levels with string description fails.")
        columns = bilocalSDP.build_columns("local2", max_monomial_length=2)
        diff = set(tuple(col) for col in columns
                   ).difference(
                       set(tuple(col)
                           for col in self.actual_cols))
        self.assertTrue(len(diff) == 0,
                        "Parsing local levels with string description fails.")
        columns = bilocalSDP.build_columns("local221", max_monomial_length=2)
        diff = set(tuple(col) for col in columns
                   ).difference(
                       set(tuple(col)
                           for col in self.actual_cols[:-1]))
        self.assertTrue(len(diff) == 0,
                        "Parsing local levels with individual string " +
                        "descriptions fails.")
        columns = bilocalSDP.build_columns("physical2",
                                           max_monomial_length=2)
        physical = (self.actual_cols[:22] + [self.actual_cols[24]]
                    + [self.actual_cols[26]] + self.actual_cols[32:])
        diff = set(tuple(col) for col in columns
                   ).difference(
                       set(tuple(col)
                           for col in physical))
        self.assertTrue(len(diff) == 0,
                        "Parsing physical levels with global string " +
                        "description fails.")
        columns = bilocalSDP.build_columns("physical", max_monomial_length=2)
        diff = set(tuple(col) for col in columns
                   ).difference(
                       set(tuple(col)
                           for col in physical))
        self.assertTrue(len(diff) == 0,
                        "Parsing physical levels without further " +
                        "description fails.")
        columns = bilocalSDP.build_columns("physical121",
                                           max_monomial_length=2)
        diff = set(tuple(col) for col in columns
                   ).difference(
                       set(tuple(col)
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
        truth    = [np.array([[1, 1, 0, 0]]),
                    np.array([[1, 1, 0, 0], [1, 1, 1, 0]]),
                    np.array([[1, 1, 1, 0], [1, 1, 0, 0]]),
                    np.array([[1, 1, 1, 0]])]
        truth    = [np.empty((0,4), dtype=int)] + [oneParty.mon_to_lexrepr(mon) 
                                                   for mon in truth]
        self.assertTrue(len(columns) == len(truth),
                        "Generating columns with identities is not producing "
                        + "the correct number of columns.")
        areequal = all(np.array_equiv(r[0].T, np.array(r[1]).T)
                       for r in zip(columns, truth))
        self.assertTrue(areequal,
                        "The column generation is not capable of handling " +
                        "monomials that reduce to the identity" +
                        f"\ngot: {columns} \nexpected: {truth}")


class TestReset(unittest.TestCase):
    sdp = InflationSDP(trivial)
    sdp.generate_relaxation("npa1")
    physical_bounds = {m: 0. for m in sdp.hermitian_moments}
    del physical_bounds[sdp.One]

    def prepare_objects(self, infSDP):
        var1 = infSDP.measurements[0][0][0][0]
        var2 = infSDP.measurements[0][0][1][0]
        infSDP.set_objective(var1, "max")
        infSDP.set_bounds({var1*var2: 0.9}, "up")
        infSDP.set_bounds({var1*var2: 0.1}, "lo")
        infSDP.set_values({var2: 0.5})

    def test_reset_all(self):
        self.prepare_objects(self.sdp)
        self.sdp.reset("all")
        self.assertEqual(self.sdp.moment_lowerbounds,
                         self.physical_bounds,
                         "Resetting lower bounds fails.")
        self.assertEqual(self.sdp.moment_upperbounds, dict(),
                         "Resetting processed upper bounds fails.")
        self.assertEqual(self.sdp.objective, {self.sdp.One: 0.},
                         "Resetting the objective function fails.")
        self.assertEqual(self.sdp.semiknown_moments, dict(),
                         "Resetting the known values fails to empty " +
                         "semiknown_moments.")
        self.assertEqual(self.sdp.known_moments, {self.sdp.One: 1.},
                         "Resetting the known values fails to empty " +
                         "known_moments.")

    def test_reset_bounds(self):
        self.prepare_objects(self.sdp)
        correct = {key: val for key, val in self.physical_bounds.items()
                   if key not in self.sdp.known_moments}
        self.sdp.reset("bounds")
        self.assertEqual(self.sdp.moment_lowerbounds,
                         correct,
                         "Resetting lower bounds fails.")
        self.assertEqual(self.sdp.moment_upperbounds, dict(),
                         "Resetting upper bounds fails.")
        self.assertTrue(len(self.sdp.objective) == 2,
                        "Resetting the bounds resets the objective function.")
        self.assertTrue(len(self.sdp.known_moments) == 3,
                        "Resetting the bounds resets the known moments.")

    def test_reset_some(self):
        self.prepare_objects(self.sdp)
        self.sdp.reset(["objective", "values"])
        self.assertEqual(self.sdp.objective, {self.sdp.One: 0.},
                         "Resetting the objective function fails.")
        self.assertEqual(self.sdp.semiknown_moments, dict(),
                        "Resetting the known values fails to empty " +
                        "semiknown_moments.")
        self.assertEqual(self.sdp.known_moments, {self.sdp.One: 1.},
                         "Resetting the known values fails to empty " +
                         "known_moments.")
        self.assertTrue(len(self.sdp.moment_lowerbounds) == 4,
                        "Lower bounds are being reset when they should not.")
        self.assertTrue(len(self.sdp.moment_upperbounds) == 1,
                        "Upper bounds are being reset when they should not.")

    def test_reset_objective(self):
        self.prepare_objects(self.sdp)
        self.sdp.reset("objective")
        self.assertEqual(self.sdp.objective, {self.sdp.One: 0.},
                         "Resetting the objective function fails.")
        self.assertTrue(len(self.sdp.known_moments) == 3,
                        "Resetting the objective resets the known moments.")
        self.assertTrue(len(self.sdp.moment_lowerbounds) == 4,
                        "Resetting the objective resets the lower bounds.")
        self.assertTrue(len(self.sdp.moment_upperbounds) == 1,
                        "Resetting the objective resets the upper bounds.")

    def test_reset_values(self):
        self.prepare_objects(self.sdp)
        self.sdp.reset("values")
        self.assertEqual(self.sdp.semiknown_moments, dict(),
                         "Resetting the known values fails to empty " +
                         "semiknown_moments.")
        self.assertEqual(self.sdp.known_moments, {self.sdp.One: 1.},
                         "Resetting the known values fails to empty " +
                         "known_moments.")
        self.assertTrue(len(self.sdp.moment_lowerbounds) == 4,
                        "Resetting the objective resets the lower bounds.")
        self.assertTrue(len(self.sdp.moment_upperbounds) == 1,
                        "Resetting the objective resets the upper bounds.")
        self.assertTrue(len(self.sdp.objective) == 2,
                        "Resetting the bounds resets the objective function.")


class TestResetLP(unittest.TestCase):
    lp = InflationLP(trivial_c, nonfanout=False)
    lp._generate_lp()

    def setUp(self) -> None:
        var1 = "P[v_0=0 v_1=0]"
        var2 = "pv(0|1)"
        self.lp.set_objective({var1: 1}, "max")
        self.lp.set_bounds({var1: 0.9}, "up")
        self.lp.set_bounds({var1: 0.1}, "lo")
        self.lp.set_values({var2: 0.5})

    def test_reset_all(self):
        self.lp.reset("all")
        self.assertEqual(self.lp.moment_lowerbounds, dict(),
                         "Resetting lower bounds failed.")
        self.assertEqual(self.lp.moment_upperbounds, dict(),
                         "Resetting upper bounds failed.")
        self.assertEqual(self.lp.objective, dict(),
                         "Resetting objective failed.")
        self.assertEqual(self.lp.semiknown_moments, dict(),
                         "Resetting known values failed to empty "
                         "semiknown_moments.")
        self.assertEqual(self.lp.known_moments, {self.lp.One: 1.},
                         "Resetting known values failed to empty "
                         "known_moments.")


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
    bellScenario_c = InflationProblem({"Lambda": ["A", "B"]},
                                      outcomes_per_party=[2, 2],
                                      settings_per_party=[2, 2],
                                      inflation_level_per_source=[1],
                                      classical_sources='all')

    cutInflation = InflationProblem({"lambda": ["a", "b"],
                                     "mu": ["b", "c"],
                                     "sigma": ["a", "c"]},
                                    outcomes_per_party=[2, 2, 2],
                                    settings_per_party=[1, 1, 1],
                                    inflation_level_per_source=[2, 1, 1])

    cutInflation_c = InflationProblem({"lambda": ["a", "b"],
                                       "mu": ["b", "c"],
                                       "sigma": ["a", "c"]},
                                      outcomes_per_party=[2, 2, 2],
                                      settings_per_party=[1, 1, 1],
                                      inflation_level_per_source=[2, 1, 1],
                                      classical_sources="all")

    instrumental = InflationProblem({"U_AB": ["A", "B"],
                                     "A": ["B"]},
                                    outcomes_per_party=(2, 2),
                                    settings_per_party=(3, 1),
                                    inflation_level_per_source=(1,),
                                    order=("A", "B"))
    instrumental_c = InflationProblem({"U_AB": ["A", "B"],
                                       "A": ["B"]},
                                      outcomes_per_party=(2, 2),
                                      settings_per_party=(3, 1),
                                      inflation_level_per_source=(1,),
                                      order=("A", "B"),
                                      classical_sources='all')

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

    def test_CHSH(self):
        sdp = InflationSDP(self.bellScenario)
        sdp.generate_relaxation("npa1")
        self.assertEqual(sdp.n_columns, 5,
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
        equalities = sdp.minimal_equalities

        self.assertEqual(len(equalities), 738,
                         "Failing to obtain the correct number of implicit " +
                         "equalities in a non-network scenario.")

        # TODO: When we add support for user-specifiable equalities, modify
        # this test to only check implicit equalities.
        self.assertTrue(all(set(equality.values()) == {-1, 1}
                            for equality in equalities),
                        "Some implicit equalities lack a nontrivial left-hand"
                        + "or right-hand side.")

        self.assertTrue(all(sdp.Zero.name not in equality.keys()
                            for equality in equalities),
                        "Some implicit equalities are wrongly assigning " +
                        "coefficients to the zero monomial.")

    def test_GHZ_commuting(self):
        sdp = InflationSDP(self.cutInflation_c)
        sdp.generate_relaxation("local1")
        self.assertEqual(sdp.n_columns, 18,
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
        lp_fanout = InflationLP(self.cutInflation_c, nonfanout=False)
        lp_fanout.set_distribution(self.GHZ(0.5 + 1e-2))
        lp_fanout.solve()
        self.assertEqual(lp_fanout.success, False,
                         "The fanout LP is not identifying incompatible " +
                         "distributions.")
        lp_nonfanout = InflationLP(self.cutInflation_c, nonfanout=True)
        lp_nonfanout.set_distribution(self.GHZ(0.5 + 1e-2))
        lp_nonfanout.solve()
        self.assertEqual(lp_nonfanout.success, False,
                         "The nonfanout LP is not identifying incompatible " +
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
        lp_fanout.set_distribution(self.GHZ(0.5 - 1e-2))
        lp_fanout.solve()
        self.assertEqual(lp_fanout.success, True,
                         "The fanout LP is not recognizing compatible " +
                         "distributions.")
        lp_nonfanout.set_distribution(self.GHZ(0.5 - 1e-2))
        lp_nonfanout.solve()
        self.assertEqual(lp_nonfanout.success, True,
                         "The nonfanout LP is not identifying incompatible " +
                         "distributions.")
        sdp.solve(feas_as_optim=True)
        self.assertTrue(sdp.primal_objective >= 0,
                        "The commuting SDP with feasibility as optimization " +
                        "is not recognizing compatible distributions.")

    def test_GHZ_NC(self):
        sdp = InflationSDP(self.cutInflation)
        sdp.generate_relaxation("local1")
        self.assertEqual(sdp.n_columns, 18,
                         "The number of generating columns is not correct.")
        self.assertEqual(sdp.n_knowable, 8 + 1,  # only '1' is included here.
                         "The count of knowable moments is wrong.")
        self.assertEqual(sdp.n_unknowable, 13,
                         "The count of unknowable moments is wrong.")

        sdp.set_distribution(self.GHZ(0.5 + 1e-2))
        self.assertTrue(np.isclose(sdp.known_moments[sdp.moments[8]],
                                   (0.5+1e-2) / 2 + (0.5-1e-2) / 8),
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
        self.assertTrue(np.isclose(sdp.known_moments[sdp.moments[8]],
                                   (0.5-1e-2) / 2 + (0.5+1e-2) / 8),
                        "Re-setting the distribution is failing.")
        sdp.solve()
        self.assertEqual(sdp.status, "feasible",
                         "The non-commuting SDP is not recognizing " +
                         "compatible distributions.")
        sdp.solve(feas_as_optim=True)
        self.assertTrue(sdp.primal_objective >= 0,
                        "The NC SDP with feasibility as optimization is not " +
                        "recognizing compatible distributions.")

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

    def test_lpi(self):
        sdp = InflationSDP(trivial)
        [[[[A10], [A11]], [[A20], [A21]]]] = sdp.measurements
        sdp.generate_relaxation([1,
                                 A10, A11, A20, A21,
                                 A10*A11, A10*A21, A11*A20, A20*A21])
        sdp.set_distribution(np.array([[0.14873, 0.85168]]))
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
        cols = [np.array([]),
                np.array([[1, 2, 0, 0],
                 [1, 2, 1, 0]]),
                np.array([[1, 1, 0, 0],
                 [1, 2, 0, 0],
                 [1, 2, 1, 0]])]
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

    def test_new_indices(self):
        sdp  = InflationSDP(trivial)
        cols = [np.array([]),
                np.array([[1, 1, 0, 0],
                          [1, 2, 0, 0],
                          [1, 2, 1, 0]])]
        sdp.generate_relaxation(cols)
        sdp.set_distribution(np.ones((2, 1)) / 2,
                             use_lpi_constraints=True)
        new_mon_indices = np.array([semi[1][1].idx
                                    for semi in sdp.semiknown_moments.items()])
        self.assertTrue(np.all(new_mon_indices > len(sdp.moments)),
                        "The new unknown monomials that appear when applying" +
                        " LPI constraints are not assigned correct indices.")

    def test_supports(self):
        sdp = InflationSDP(self.bellScenario, supports_problem=True)
        sdp.generate_relaxation("local1")
        pr_support = np.zeros((2, 2, 2, 2))
        for a, b, x, y in np.ndindex(*pr_support.shape):
            if x*y == (a + b) % 2:
                pr_support[a, b, x, y] = np.random.randn()
        sdp.set_distribution(pr_support)
        sdp.solve(feas_as_optim=False)
        self.assertEqual(sdp.status, "infeasible",
                         "Failing to detect the infeasibility of a support "
                         + "known to be incompatible.")
        compatible_support = np.ones((2, 2, 2, 2), dtype=float)
        sdp.set_distribution(compatible_support)
        sdp.solve(feas_as_optim=False)
        self.assertEqual(sdp.status, "feasible",
                         "A feasible support for the Bell scenario is not " +
                         "being recognized as such.")


class TestLPOutput(unittest.TestCase):
    def test_bounds(self):
        ub = 0.8
        lb = 0.2
        very_trivial = InflationProblem({"a": ["b"]}, outcomes_per_party=[2])
        lp = InflationLP(very_trivial)
        operator = np.asarray(lp.monomials).flatten()[1]
        with self.subTest(msg="Setting upper bound"):
            lp.set_objective({operator: 1}, "max")
            lp.set_bounds({operator: ub}, "up")
            lp.solve()
            self.assertAlmostEqual(
                lp.objective_value, ub,
                msg=f"Setting upper bounds to monomials failed. The result "
                    f"obtained for {{max x s.t. x <= {ub}}} is "
                    f"{lp.objective_value}.")
        with self.subTest(msg="Setting lower bound"):
            lp.set_objective({operator: 1}, "min")
            lp.set_bounds({operator: lb}, "lo")
            lp.solve()
            self.assertAlmostEqual(
                lp.objective_value, lb,
                msg=f"Setting lower bounds to monomials failed. The result "
                    f"obtained for {{min x s.t. x >= {lb}}} is "
                    f"{lp.objective_value}.")

    def test_instrumental(self):
        lp = InflationLP(TestSDPOutput.instrumental_c, nonfanout=False)
        with self.subTest(msg="Infeasible Bonet's inequality"):
            lp.set_distribution(TestSDPOutput.incompatible_dist)
            lp.solve(feas_as_optim=False)
            self.assertTrue(lp.status in ["prim_infeas_cer", "dual_infeas_cer",
                                          "unknown"],
                            "Failed to detect the infeasibility of the "
                            "distribution that maximally violates Bonet's "
                            "inequality.")
        with self.subTest(msg="Infeasible normalization"):
            unnormalized_dist = np.ones((2, 2, 3, 1), dtype=float)
            lp.set_distribution(unnormalized_dist)
            lp.solve(feas_as_optim=False)
            self.assertTrue(lp.status in ["prim_infeas_cer", "dual_infeas_cer",
                                          "unknown"],
                            "Failed to detect the infeasibility of a "
                            "distribution that violates normalization.")
        with self.subTest(msg="Feasible instrumental"):
            compatible_dist = unnormalized_dist / 4
            lp.set_distribution(compatible_dist)
            lp.solve(feas_as_optim=False)
            self.assertEqual(lp.status, "optimal",
                             "A feasible distribution for the instrumental "
                             "scenario is not being recognized as such.")

    def test_supports(self):
        lp = InflationLP(TestSDPOutput.bellScenario_c, supports_problem=True,
                         nonfanout=False)
        with self.subTest(msg="Incompatible support"):
            pr_support = np.zeros((2, 2, 2, 2))
            for a, b, x, y in np.ndindex(*pr_support.shape):
                if x*y == (a + b) % 2:
                    pr_support[a, b, x, y] = np.random.randn()
            lp.set_distribution(pr_support)
            lp.solve(feas_as_optim=False)
            self.assertIn(lp.status, ["prim_infeas_cer", "dual_infeas_cer"],
                          "Failed to detect the infeasibility of a support "
                          "known to be incompatible.")
        with self.subTest(msg="Compatible support"):
            compatible_support = np.ones((2, 2, 2, 2), dtype=float)
            lp.set_distribution(compatible_support)
            lp.solve(feas_as_optim=False)
            self.assertEqual(lp.status, "optimal",
                             "A feasible support for the Bell scenario was "
                             "not recognized as such.")


class TestSymmetries(unittest.TestCase):
    def test_apply_symmetries(self):
        from inflation.sdp.quantum_tools import \
                                                apply_inflation_symmetries
        G = np.array([[0,  1,  2,  3,  4,  5],
                      [6,  7,  8,  9, 10, 11],
                      [12, 13, 14, 15, 16, 17],
                      [18, 19, 20, 21, 22, 23],
                      [24, 25, 26, 27, 28, 29],
                      [30, 31, 32, 33, 34, 35]])
        # The symmetries are that equal variables are (1,2), (3, 4, 5), (8, 9)
        sym_mm, orbits, repr_values = \
            apply_inflation_symmetries(G, np.array([[0, 1, 2, 3, 5, 4],
                                                    [0, 2, 3, 1, 4, 5]]))
        sym_mm_good = np.array([[0,  1,  2,  1,  3,  3],
                                [4,  5,  6,  7,  8,  8],
                                [9, 10, 11, 12, 13, 13],
                                [4,  6,  7,  5,  8,  8],
                                [14, 15, 16, 15, 17, 18],
                                [14, 15, 16, 15, 18, 17]])
        self.assertTrue(np.allclose(sym_mm, sym_mm_good),
                        "Symmetrized moment matrix is not correct.")
        orbits_good = {0: 0, 1: 1, 2: 2, 3: 1, 4: 3, 5: 3, 6: 4, 7: 5, 8: 6,
                       9: 7, 10: 8, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12,
                       16: 13, 17: 13, 18: 4, 19: 6, 20: 7, 21: 5, 22: 8,
                       23: 8, 24: 14, 25: 15, 26: 16, 27: 15, 28: 17, 29: 18,
                       30: 14, 31: 15, 32: 16, 33: 15, 34: 18, 35: 17}
        self.assertTrue(orbits == orbits_good,
                        "Orbits dictionary is not correct.")
        repr_values_good = np.array([0,  1,  2,  4,  6,  7,  8,  9, 10,
                                     12, 13, 14, 15, 16, 24, 25, 26, 28, 29])
        self.assertTrue(np.allclose(repr_values, repr_values_good),
                        "Representatives mapping is not correct.")

    def test_commutations_after_symmetrization(self):
        scenario = InflationSDP(trivial_c)
        col_structure = [np.array([]),
                         np.array([[1, 2, 0, 0], [1, 2, 1, 0]]),
                         np.array([[1, 1, 0, 0], [1, 2, 0, 0]]),
                         np.array([[1, 1, 1, 0], [1, 2, 0, 0]]),
                         np.array([[1, 1, 0, 0], [1, 2, 1, 0]]),
                         np.array([[1, 1, 1, 0], [1, 2, 1, 0]]),
                         np.array([[1, 1, 0, 0], [1, 1, 1, 0]])]

        scenario.generate_relaxation(col_structure)
        self.assertTrue(np.array_equal(scenario.columns_symmetries,
                                       [[0, 6, 2, 4, 3, 5, 1]]),
                        "The commutation relations of different copies are " +
                        "not applied properly after inflation symmetries.")

    def test_detected_symmetries(self):
        cols = bilocalSDP.build_columns('local1')
        # bilocalSDP.generating_monomials = cols
        # bilocalSDP.generating_monomials_1d = list(map, bilocalSDP)
        # bilocalSDP.n_columns = len(cols)
        # bilocalSDP.genmon_hash_to_index = {
        #                         bilocalSDP._from_2dndarray(op): i
        #                         for i, op in enumerate(cols)}
        syms = bilocalSDP._discover_columns_symmetries()
        # Make it a set so the order doesn't matter
        syms = set(tuple(s) for s in syms)
        # I simply copied the output at a time when we understand the code
        # to be working; this test simply detects if the code changes
        syms_good = {(0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14,
                      13, 16, 15, 18, 17, 20, 19, 24, 23, 22, 21, 28,
                      27, 26, 25, 32, 31, 30, 29, 36, 35, 34, 33, 40,
                      39, 38, 37, 44, 43, 42, 41),
                     (0, 2, 1, 5, 6, 3, 4, 7, 8, 15, 16, 13, 14, 11,
                      12, 9, 10, 19, 20, 17, 18, 25, 26, 27, 28, 21,
                      22, 23, 24, 41, 42, 43, 44, 37, 38, 39, 40, 33,
                      34, 35, 36, 29, 30, 31, 32),
                     (0, 2, 1, 6, 5, 4, 3, 8, 7, 16, 15, 14, 13, 12,
                      11, 10, 9, 20, 19, 18, 17, 28, 27, 26, 25, 24,
                      23, 22, 21, 44, 43, 42, 41, 40, 39, 38, 37, 36,
                      35, 34, 33, 32, 31, 30, 29)}
        self.assertEqual(syms, syms_good, "The symmetries are not being " +
                                          "detected correctly.")


class TestConstraintGeneration(unittest.TestCase):
    def test_norm_eqs_mon2index_mapping(self):
        sdp = InflationSDP(InflationProblem({'r': ['A']}, (3,), (3,), (3,)))
        sdp.generate_relaxation('npa2')
        for i, lexmon in enumerate(sdp.generating_monomials_1d):
            self.assertTrue(i == sdp.genmon_1d_to_index[tuple(lexmon)],
                            "Monomials in the generating list must be mapped "
                            + "to their index in the list under "
                            + "InflationSDP.genmon_hash_to_index for "
                            + "InflationSDP._discover_normalization_eqns() "
                            + "to work correctly.")

    def test_norm_eqs_expansion(self):
        from inflation.sdp.quantum_tools import \
                                                expand_moment_normalisation

        out = expand_moment_normalisation(np.array([[1, 1, 0, 1]]),
                                          [3],
                                          {i: False for i in range(3)})
        out_good = [(np.array([]).reshape((0, 4)),
                    [np.array([[1, 1, 0, 1]]), np.array([[1, 1, 0, 0]])])]
        for eq1, eq2 in zip(out, out_good):
            self.assertTrue(np.allclose(eq1[0], eq2[0])
                            and len(eq1[1]) == len(eq2[1]),
                            "Normalisation constraint is not" +
                            "being properly generated. ")
            for el1, el2 in zip(eq1[1], eq2[1]):
                self.assertTrue(np.allclose(el1, el2),
                                "Normalisation constraint is not" +
                                "being properly generated. ")

        # Note, currently if more than one operator has a last output,
        # several equalities will be generated, instead of a single one.
        out = expand_moment_normalisation(np.array([[1, 1, 0, 1],
                                                    [2, 3, 1, 1],
                                                    [2, 3, 1, 0]]),
                                          [3, 3, 2],  # we 'lie' about card.
                                          {i: False for i in range(3)})
        out_good = [(np.array([[2, 3, 1, 1], [2, 3, 1, 0]]),
                    [np.array([[1, 1, 0, 1], [2, 3, 1, 1], [2, 3, 1, 0]]),
                     np.array([[1, 1, 0, 0], [2, 3, 1, 1], [2, 3, 1, 0]])]),
                    (np.array([[1, 1, 0, 1], [2, 3, 1, 0]]),
                    [np.array([[1, 1, 0, 1], [2, 3, 1, 1], [2, 3, 1, 0]]),
                     np.array([[1, 1, 0, 1], [2, 3, 1, 0], [2, 3, 1, 0]])])]
        for eq1, eq2 in zip(out, out_good):
            self.assertTrue(np.allclose(eq1[0], eq2[0])
                            and len(eq1[1]) == len(eq2[1]),
                            "Normalisation constraint is not" +
                            "being properly generated. ")
            for el1, el2 in zip(eq1[1], eq2[1]):
                self.assertTrue(np.allclose(el1, el2),
                                "Normalisation constraint is not" +
                                "being properly generated.")

    def test_normeqs_colineq2momentineq(self):
        from inflation.sdp.quantum_tools import \
                                            construct_normalization_eqs
        G = np.array([[1,  2,  3,  4,  5],
                      [2,  1,  6,  7,  8],
                      [3,  6,  1,  9, 10],
                      [4,  7,  9,  1, 11],
                      [5,  8, 10, 11,  1]])
        # The following inequalities have random integers, they have no
        # interpretation
        column_inequalities = [(0, [1, 2, 3, 4]), (3, [2, 4])]
        out = construct_normalization_eqs(column_inequalities, G)
        out_good = [(1, [2, 3, 4, 5]),
                    (2, [1, 6, 7, 8]),
                    (3, [1, 6, 9, 10]),
                    (4, [1, 7, 9, 11]),
                    (5, [1, 8, 10, 11]),
                    (4, [3, 5]),
                    (7, [6, 8]),
                    (9, [1, 10]),
                    (1, [9, 11])]
        self.assertEqual(out,
                         out_good,
                         "Column equalities are not being " +
                         "properly lifted to moment inequalities.")


class TestPipelineLP(unittest.TestCase):
    def _monomial_generation(self, **args):
        with self.subTest(msg="Testing monomial generation"):
            raw_n_columns = args["scenario"].raw_n_columns
            truth = args["truth_columns"]
            self.assertEqual(raw_n_columns, truth,
                             f"There are {raw_n_columns} columns but {truth} "
                             f"were expected.")

    def _equalities(self, **args):
        with self.subTest(msg="Testing equalities"):
            equalities = args["scenario"].moment_equalities
            truth = args["truth_eq"]
            self.assertEqual(len(equalities), truth,
                             f"There are {len(equalities)} equalities but "
                             f"{truth} were expected.")
            self.assertTrue(all(set(equality.values()) == {-1, 1}
                                for equality in equalities),
                            "Some implicit equalities lack a nontrivial "
                            "left-hand or right-hand side.")

    def _test_a_visibility(self, **args):
        lp = args["scenario"]
        dist_func = args["dist_func"]
        dist_name = args["dist_name"]
        crit_cutoff = args["crit_cutoff"]
        with self.subTest(msg="Testing GHZ, incompatible distribution"):
            lp.set_distribution(dist_func(crit_cutoff + 1e-2))
            lp.solve()
            self.assertIn(lp.status, ["prim_infeas_cer", "dual_infeas_cer",
                                      "unknown"],
                          "The LP did not identify the incompatible "
                          "distribution.")
        with self.subTest(msg=f"Testing {dist_name}, incompatible distribution, "
                              "feasibility as optimization"):
            lp.solve(feas_as_optim=True)
            self.assertTrue(lp.primal_objective <= 0,
                            "The LP with feasibility as optimization did not "
                            "identify the incompatible distribution.")
        with self.subTest(msg=f"Testing {dist_name}, compatible distribution"):
            lp.set_distribution(dist_func(crit_cutoff - 1e-2))
            lp.solve()
            self.assertEqual(lp.status, "optimal",
                             "The LP did not recognize the compatible "
                             "distribution.")
        with self.subTest(msg=f"Testing {dist_name}, compatible distribution, "
                              "feasibility as optimization"):
            # self.skipTest("Feasibility as optimization not working for "
            #               "compatible distributions?")
            lp.solve(feas_as_optim=True)
            self.assertTrue(lp.primal_objective >= -1e-6,
                            "The LP with feasibility as optimization did not "
                            "recognize the compatible distribution.")

    def _run(self, **args):
        self._monomial_generation(**args)
        self._equalities(**args)
        self._test_a_visibility(**args)


class TestInstrumental(TestPipelineLP):
    instrumental = InflationProblem({"U_AB": ["A", "B"],
                                     "A": ["B"]},
                                    outcomes_per_party=(2, 2),
                                    settings_per_party=(2, 1),
                                    inflation_level_per_source=(1,),
                                    order=("A", "B"))
    instrumental_c = InflationProblem({"U_AB": ["A", "B"],
                                       "A": ["B"]},
                                      outcomes_per_party=(2, 2),
                                      settings_per_party=(2, 1),
                                      inflation_level_per_source=(1,),
                                      order=("A", "B"),
                                      classical_sources='all')

    def p_Pearl_Violating(self, v):
        dist = np.full((2, 2, 2, 1), (1 - v) / 4)
        dist[0, 0, 0, 0] = dist[0, 1, 1, 0] = v + (1 - v) / 4
        return dist

    def test_instrumental_fanout(self):
        inst = InflationLP(self.instrumental_c, nonfanout=False)
        args = {"scenario": inst,
                "truth_columns": 36,
                "truth_eq": 20,
                "dist_func": self.p_Pearl_Violating,
                "dist_name": "Instrumental noisy e-sep violating",
                "crit_cutoff": 1/3}
        self._run(**args)

    def test_instrumental_nonfanout(self):
        inst = InflationLP(self.instrumental, nonfanout=True)
        args = {"scenario": inst,
                "truth_columns": 15,
                "truth_eq": 6,
                "dist_func": self.p_Pearl_Violating,
                "dist_name": "Instrumental noisy e-sep violating",
                "crit_cutoff": 1/3}
        self._run(**args)


class TestBell(TestPipelineLP):
    bellScenario = InflationProblem({"Lambda": ["A", "B"]},
                                    outcomes_per_party=[2, 2],
                                    settings_per_party=[2, 2],
                                    inflation_level_per_source=[1])
    bellScenario_c = InflationProblem({"Lambda": ["A", "B"]},
                                      outcomes_per_party=[2, 2],
                                      settings_per_party=[2, 2],
                                      inflation_level_per_source=[1],
                                      classical_sources='all')

    def _CHSH(self, **args):
        lp = args["scenario"]
        truth = args["truth_obj"]
        lp.set_objective({1: 2.0,
                          "pA(0|0)": -4.0,
                          "pB(0|0)": -4.0,
                          "pAB(00|11)": -4.0,
                          "pAB(00|00)": 4.0,
                          "pAB(00|01)": 4.0,
                          "pAB(00|10)": 4.0}, "max")
        with self.subTest(msg="Testing max CHSH"):
            lp.solve()
            self.assertAlmostEqual(lp.objective_value, truth,
                                   msg=f"The LP is not recovering max(CHSH) = "
                                       f"{truth}.")
        # Biased CHSH?

    def p_Signalling_to_Bob(self, v):
        dist = np.full((2, 2, 2, 2), (1 - v) / 4)
        dist[0, 0, 0, 0] = dist[0, 1, 1, 0] = v + (1 - v) / 4
        return dist

    def test_bell_fanout(self):
        bell = InflationLP(self.bellScenario_c, nonfanout=False)
        args = {"scenario": bell,
                "truth_columns": 16,
                "truth_obj": 2,
                "truth_eq": 0,
                "dist_func": self.p_Signalling_to_Bob,
                "dist_name": "Noisy Signalling to Bob",
                "crit_cutoff": 0.2}
        self._run(**args)
        # self._CHSH(**args)

    def test_bell_nonfanout(self):
        bell = InflationLP(self.bellScenario, nonfanout=True)
        args = {"scenario": bell,
                "truth_columns": 9,
                "truth_obj": 2,
                "truth_eq": 0,
                "dist_func": self.p_Signalling_to_Bob,
                "dist_name": "Noisy Signalling to Bob",
                "crit_cutoff": 0.2}
        self._run(**args)
        # self._CHSH(**args)


class TestTriangle(TestPipelineLP):
    triangle = InflationProblem({"lambda": ["a", "b"],
                                 "mu": ["b", "c"],
                                 "sigma": ["a", "c"]},
                                outcomes_per_party=[2, 2, 2],
                                settings_per_party=[1, 1, 1],
                                inflation_level_per_source=[1, 1, 1],
                                order=['a', 'b', 'c'])

    def GHZ(self, v):
        dist = np.zeros((2, 2, 2, 1, 1, 1))
        dist[0, 0, 0] = dist[1, 1, 1] = 1 / 2
        return v * dist + (1 - v) / 8

    def test_triangle_fanout(self):
        triangle = InflationLP(self.triangle, nonfanout=False)
        args = {"scenario": triangle,
                "truth_columns": 8,
                "truth_eq": 0,
                "dist_func": self.GHZ,
                "dist_name": "Noisy GHZ",
                "crit_cutoff": 1}
        self._run(**args)

    def test_triangle_nonfanout(self):
        triangle = InflationLP(self.triangle, nonfanout=True)
        args = {"scenario": triangle,
                "truth_columns": 8,
                "truth_eq": 0,
                "dist_func": self.GHZ,
                "dist_name": "Noisy GHZ",
                "crit_cutoff": 1}
        self._run(**args)


class TestEvans(TestPipelineLP):
    evans = InflationProblem({"U_AB": ["A", "B"],
                              "U_BC": ["B", "C"],
                              "B": ["A", "C"]},
                             outcomes_per_party=(2, 2, 2),
                             settings_per_party=(1, 1, 1),
                             inflation_level_per_source=(1, 1),
                             order=("A", "B", "C"))
    evans_c = InflationProblem({"U_AB": ["A", "B"],
                                "U_BC": ["B", "C"],
                                "B": ["A", "C"]},
                               outcomes_per_party=(2, 2, 2),
                               settings_per_party=(1, 1, 1),
                               inflation_level_per_source=(1, 1),
                               order=("A", "B", "C"),
                               classical_sources='all')

    def p_Evans_esep_violating(self, v):
        dist = np.zeros((2, 2, 2, 1, 1, 1))
        for x, y, z in product(range(2), repeat=3):
            dist[x, y, z] = (1 + v * (-1) ** (x + y + y * z)) / 8
        return dist

    def test_evans_fanout(self):
        evans = InflationLP(self.evans_c, nonfanout=False)
        args = {"scenario": evans,
                "truth_columns": 48,
                "truth_eq": 16,
                "dist_func": self.p_Evans_esep_violating,
                "dist_name": "Noisy Evans Incompatible",
                "crit_cutoff": 1}
        self._run(**args)

    def test_evans_nonfanout(self):
        evans = InflationLP(self.evans, nonfanout=True)
        args = {"scenario": evans,
                "truth_columns": 27,
                "truth_eq": 9,
                "dist_func": self.p_Evans_esep_violating,
                "dist_name": "Noisy Evans Incompatible",
                "crit_cutoff": 1}
        self._run(**args)
        

class TestFullNN(TestPipelineLP):
    scenario = InflationProblem({'lambda': ['A', 'B'],
                                     'NS': ['B', 'C']},
                                [2, 4, 2], [3, 1, 3], 
                                inflation_level_per_source=[1, 2],
                                classical_sources=['lambda'],)

    def _prob_EJM(self, v, theta=0):
        p_bell = np.expand_dims((0, 1, -1, 0), axis=1)/np.sqrt(2)
        rho_v = v * p_bell @ p_bell.conj().T + (1 - v) * np.eye(4)/4
        sigmax = np.array([[0, 1], [1, 0]])
        sigmay = np.array([[0, -1j], [1j, 0]])
        sigmaz = np.array([[1, 0], [0, -1]])
        A = [[np.expand_dims(v, axis=1) @ np.expand_dims(v, axis=1).conj().T 
                for v in reversed(np.linalg.eigh(op)[1].T)] 
                for op in [sigmax, sigmay, sigmaz]]
        C = [[np.expand_dims(v, axis=1) @ np.expand_dims(v, axis=1).conj().T 
                for v in reversed(np.linalg.eigh(op)[1].T)] 
                for op in [sigmax, sigmay, sigmaz]]
        r_plus = (1 + np.exp(1j*theta))/np.sqrt(2)
        r_minus = (1 - np.exp(1j*theta))/np.sqrt(2)
        e00 = np.expand_dims([1, 0, 0, 0], axis=1)
        e01 = np.expand_dims([0, 1, 0, 0], axis=1)
        e10 = np.expand_dims([0, 0, 1, 0], axis=1)
        e11 = np.expand_dims([0, 0, 0, 1], axis=1)
        psi1 = 1/2 * (np.exp(-1j*np.pi/4)*e00 - r_plus * e01 
                        - r_minus * e10 + np.exp(-3/4*np.pi*1j)*e11)
        psi2 = 1/2 * (np.exp(1j*np.pi/4)*e00 + r_minus * e01 
                        + r_plus * e10 + np.exp(3/4*np.pi*1j)*e11)
        psi3 = 1/2 * (np.exp(-1j*np.pi*3/4)*e00 + r_minus * e01 
                        + r_plus * e10 + np.exp(-1/4*np.pi*1j)*e11)
        psi4 = 1/2 * (np.exp(1j*np.pi*3/4)*e00 - r_plus * e01 
                        - r_minus * e10 + np.exp(np.pi*1j/4)*e11)
        B = [ [psi @ psi.conj().T for psi in [psi1, psi2, psi3, psi4] ]]
        
        p = np.zeros((2, 4, 2, 3, 1, 3))
        state = np.kron(rho_v, rho_v)
        for a, b, c, x, y, z in np.ndindex(p.shape):
            mmnt = np.kron(np.kron(A[x][a], B[y][b]), C[z][c])
            p[a, b, c, x, y, z] = np.real(np.trace(state @ mmnt))
        return p
    
    def test_fullnetworknonlocality_3partite_line(self):
        lp = InflationLP(self.scenario, nonfanout=True)
        
        best_theta = np.arccos(np.sqrt(5) / 3)
        v_for_best_theta = 2 / np.sqrt(5)
        
        args = {"scenario": lp,
                "truth_columns": 2048,
                "truth_eq": 0,
                "dist_func": lambda x: self._prob_EJM(x, theta=best_theta),
                "dist_name": "Noisy EJM",
                "crit_cutoff": v_for_best_theta}
        self._run(**args)