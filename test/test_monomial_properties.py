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

scenario_c = InflationProblem({"lambda": ["A", "B"],
                               "sigma": ["A", "C"],
                               "mu": ["B", "C"]},
                              outcomes_per_party=[2, 2, 2],
                              settings_per_party=[1, 1, 1],
                              inflation_level_per_source=[3, 2, 2],
                              order=('A', 'B', 'C'),
                              classical_sources="all")

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
        self.assertEqual(self.sdp._sanitise_moment(initial),
                         self.sdp._sanitise_moment(correct),
                         "to_canonical has problems with bringing monomials " +
                         "to representative form.")
        
    def test_commutation_with_hybrid_sources(self):
        from inflation.sdp.fast_npa import nb_operators_commute
        
        # Two source scenario. Two operators with overlap on the first source, 
        # with the first source being classical.
        # No intermediate latents.
        pairwise_source_info = np.zeros((2, 2, 2), dtype=bool)
        for i in range(2):
            pairwise_source_info[i, i] = True
        self.assertFalse(nb_operators_commute(
            np.array([1, 1, 1, 0, 0]),
            np.array([1, 1, 2, 0, 0]),
            np.array([1, 1], dtype=bool),  # both sources quantum
            sources_to_check_for_pairwise=pairwise_source_info),
            "nb_operators_commute fails to identify " +
            "non-commutativty when overlapping on quantum sources.")
        self.assertTrue(nb_operators_commute(
            np.array([1, 1, 1, 0, 0]),
            np.array([1, 1, 2, 0, 0]),
            np.array([0, 1], dtype=bool),  # first source classical
            sources_to_check_for_pairwise=pairwise_source_info),
            "nb_operators_commute fails to identify " +
            "commutativity when overlapping on classical sources.")
        self.assertTrue(nb_operators_commute(
            np.array([1, 1, 1, 0, 0]),
            np.array([1, 1, 2, 1, 0]),
            np.array([0, 1], dtype=bool),  # first source classical
            sources_to_check_for_pairwise=pairwise_source_info),
            "nb_operators_commute fails to identify " +
            "commutativity when overlapping on classical sources " +
            "with different settings for the operators.")
        self.assertTrue(nb_operators_commute(
            np.array([1, 1, 1, 0, 0]),
            np.array([1, 1, 1, 1, 0]),
            np.array([0, 0], dtype=bool),  # both sources classical
            sources_to_check_for_pairwise=pairwise_source_info),
            "nb_operators_commute fails to identify " +
            "commutativity when overlapping on classical sources " +
            "with different settings for the operators.")
        


    def test_ordering_parties(self):
        initial = 'A_1_1_0_0_0*A_1_2_0_0_0*C_0_2_1_0_0*B_1_0_2_0_0'
        correct = 'A_1_1_0_0_0*A_1_2_0_0_0*B_1_0_2_0_0*C_0_2_1_0_0'
        self.assertEqual(self.sdp._sanitise_moment(initial),
                         self.sdp._sanitise_moment(correct),
                         "to_canonical fails to order parties correctly.")


class TestToRepr(unittest.TestCase):
    sdp_commuting    = InflationSDP(scenario_c)
    sdp_noncommuting = InflationSDP(scenario)
    names = sdp_commuting.names

    def test_commuting(self):
        initial = 'A_3_1_0_0_0*A_2_1_0_0_0*A_3_1_0_0_0'
        correct = 'A_1_1_0_0_0*A_2_1_0_0_0'
        self.assertEqual(self.sdp_commuting._sanitise_moment(initial),
                         self.sdp_commuting._sanitise_moment(correct),
                         "Applying commutations to representative form fails.")

    def test_jump_sources(self):
        initial = 'A_3_1_0_0_0*A_2_1_0_0_0*A_3_1_0_0_0'
        correct = 'A_1_1_0_0_0*A_2_1_0_0_0*A_1_1_0_0_0'
        self.assertEqual(self.sdp_noncommuting._sanitise_moment(initial),
                         self.sdp_noncommuting._sanitise_moment(correct),
                         "Skipping inflation indices when computing " +
                         "representatives fails.")

    def test_swap_all_sources(self):
        initial = 'A_1_1_0_0_0*A_2_2_0_0_0*B_2_0_1_0_0'
        correct = 'A_1_1_0_0_0*A_2_2_0_0_0*B_1_0_1_0_0'
        self.assertEqual(self.sdp_noncommuting._sanitise_moment(initial),
                         self.sdp_noncommuting._sanitise_moment(correct),
                         "Swapping all sources and applying factorization " +
                         "commutations fails.")

    def test_swap_single_source(self):
        initial = 'A_1_1_0_0_0*A_2_2_0_0_0*B_1_0_2_0_0*B_2_0_1_0_0'
        correct = 'A_1_1_0_0_0*A_2_2_0_0_0*B_1_0_1_0_0*B_2_0_2_0_0'
        self.assertEqual(self.sdp_noncommuting._sanitise_moment(initial),
                         self.sdp_noncommuting._sanitise_moment(correct),
                         "Swapping a single source fails.")
