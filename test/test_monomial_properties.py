import unittest
import numpy as np

from inflation import InflationProblem, InflationSDP
from inflation.sdp.fast_npa import nb_is_knowable as is_knowable
from inflation.sdp.fast_npa import nb_is_physical as is_physical


scenario = InflationProblem({"lambda": ["A", "B"],
                             "sigma": ["A", "C"],
                             "mu": ["B", "C"]},
                            outcomes_per_party=[2, 2, 2],
                            settings_per_party=[1, 1, 1],
                            inflation_level_per_source=[3, 2, 2],
                            order=('A', 'B', 'C'))


class TestKnowable(unittest.TestCase):
    def test_knowable_complete(self):
        A12B23C31 = np.array([[1, 2, 0, 1, 0, 0],
                              [2, 2, 3, 0, 0, 0],
                              [3, 0, 3, 1, 0, 0]])

        self.assertTrue(is_knowable(A12B23C31),
                        "Monomials composed of a complete copy of the scenario"
                        + " are not identified as knowable.")

    def test_knowable_partial(self):
        A11 = np.array([[1, 1, 0, 1, 0, 0]])

        self.assertTrue(is_knowable(A11),
                        "Knowable monomials composed of a subset of parties "
                        + "are not identified as knowable.")

    def test_unknowable_complete(self):
        A11B12C22 = np.array([[1, 1, 0, 1, 0, 0],
                              [2, 1, 2, 0, 0, 0],
                              [3, 0, 2, 2, 0, 0]])

        self.assertFalse(is_knowable(A11B12C22),
                         "Monomials composed of an open copy of the scenario" +
                         " are not identified as unknowable.")

    def test_unknowable_partial(self):
        A11B11C1211 = np.array([[1, 1, 0, 1, 0, 0, 0, 0, 0],
                                [2, 1, 1, 0, 0, 0, 0, 0, 0],
                                [3, 0, 1, 2, 1, 1, 0, 0, 0]])

        self.assertFalse(is_knowable(A11B11C1211),
                         "Unknowable monomials composed of a subset of parties"
                         + " are not identified as unknowable.")


class TestPhysical(unittest.TestCase):
    def test_no_copies(self):
        # Single operator per party and knowable
        monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [3, 1, 1, 0, 0, 0]])
        self.assertEqual(is_physical(monomial), True,
                         "Physical knowable monomials are not identified as " +
                         "physical.")

        # Single operator per party but not knowable
        monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [3, 1, 2, 0, 0, 0]])
        self.assertEqual(is_physical(monomial), True,
                         "Physical unknowable monomials are not identified as "
                         + "physical.")

    def test_two_copies_physical(self):
        monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [1, 0, 2, 2, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [3, 2, 1, 0, 0, 0]])
        self.assertEqual(is_physical(monomial), True,
                         "Physical monomials with multiple copies of a party" +
                         " are not identified as such.")

    def test_two_copies_unphysical(self):
        monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [1, 0, 2, 1, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [3, 1, 2, 0, 0, 0]])
        self.assertEqual(is_physical(monomial), False,
                         "Unphysical monomials are not identified as " +
                         "unphysical.")


class TestToCanonical(unittest.TestCase):
    sdp = InflationSDP(scenario)

    def test_commutation(self):
        initial = 'A_2_1_0_0_0*A_1_2_0_0_0*B_1_0_1_0_0'
        correct = 'A_1_2_0_0_0*A_2_1_0_0_0*B_1_0_1_0_0'
        self.assertEqual(self.sdp._sanitise_monomial(initial),
                         self.sdp._sanitise_monomial(correct),
                         "to_canonical has problems with bringing monomials " +
                         "to representative form.")

    def test_ordering_parties(self):
        initial = 'A_1_1_0_0_0*A_1_2_0_0_0*C_0_2_1_0_0*B_1_0_2_0_0'
        correct = 'A_1_1_0_0_0*A_1_2_0_0_0*B_1_0_2_0_0*C_0_2_1_0_0'
        self.assertEqual(self.sdp._sanitise_monomial(initial),
                         self.sdp._sanitise_monomial(correct),
                         "to_canonical fails to order parties correctly.")


class TestToRepr(unittest.TestCase):
    sdp_commuting    = InflationSDP(scenario, commuting=True)
    sdp_noncommuting = InflationSDP(scenario, commuting=False)
    names = sdp_commuting.names

    def test_commuting(self):
        initial = 'A_3_1_0_0_0*A_2_1_0_0_0*A_3_1_0_0_0'
        correct = 'A_1_1_0_0_0*A_2_1_0_0_0'
        self.assertEqual(self.sdp_commuting._sanitise_monomial(initial),
                         self.sdp_commuting._sanitise_monomial(correct),
                         "Applying commutations to representative form fails.")

    def test_jump_sources(self):
        initial = 'A_3_1_0_0_0*A_2_1_0_0_0*A_3_1_0_0_0'
        correct = 'A_1_1_0_0_0*A_2_1_0_0_0*A_1_1_0_0_0'
        self.assertEqual(self.sdp_noncommuting._sanitise_monomial(initial),
                         self.sdp_noncommuting._sanitise_monomial(correct),
                         "Skipping inflation indices when computing " +
                         "representatives fails.")

    def test_swap_all_sources(self):
        initial = 'A_1_1_0_0_0*A_2_2_0_0_0*B_2_0_1_0_0'
        correct = 'A_1_1_0_0_0*A_2_2_0_0_0*B_1_0_1_0_0'
        self.assertEqual(self.sdp_noncommuting._sanitise_monomial(initial),
                         self.sdp_noncommuting._sanitise_monomial(correct),
                         "Swapping all sources and applying factorization " +
                         "commutations fails.")

    def test_swap_single_source(self):
        initial = 'A_1_1_0_0_0*A_2_2_0_0_0*B_1_0_2_0_0*B_2_0_1_0_0'
        correct = 'A_1_1_0_0_0*A_2_2_0_0_0*B_1_0_1_0_0*B_2_0_2_0_0'
        self.assertEqual(self.sdp_noncommuting._sanitise_monomial(initial),
                         self.sdp_noncommuting._sanitise_monomial(correct),
                         "Swapping a single source fails.")
