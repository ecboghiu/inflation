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
        pairwise_source_info_both_classical = np.zeros((2, 2, 2), dtype=np.uint8)
        pairwise_source_info_both_quantum = np.zeros((2, 2, 2), dtype=np.uint8)
        pairwise_source_info_first_classical = np.zeros((2, 2, 2), dtype=np.uint8)
        for i in range(2):
            pairwise_source_info_both_classical[i, i] = [1, 1]
            pairwise_source_info_both_quantum[i, i] = [2, 2]
            pairwise_source_info_first_classical[i, i] = [1, 2]
        self.assertFalse(nb_operators_commute(
            np.array([1, 1, 1, 0, 0]),
            np.array([1, 1, 2, 0, 0]),
            sources_to_check_for_pairwise=pairwise_source_info_both_quantum),
            "nb_operators_commute fails to identify " +
            "non-commutativty when overlapping on quantum sources.")
        self.assertTrue(nb_operators_commute(
            np.array([1, 1, 1, 0, 0]),
            np.array([1, 1, 2, 0, 0]),
            sources_to_check_for_pairwise=pairwise_source_info_first_classical),
            "nb_operators_commute fails to identify " +
            "commutativity when overlapping on classical sources.")
        self.assertTrue(nb_operators_commute(
            np.array([1, 1, 1, 0, 0]),
            np.array([1, 1, 2, 1, 0]),
            sources_to_check_for_pairwise=pairwise_source_info_first_classical),
            "nb_operators_commute fails to identify " +
            "commutativity when overlapping on classical sources " +
            "with different settings for the operators.")
        self.assertTrue(nb_operators_commute(
            np.array([1, 1, 1, 0, 0]),
            np.array([1, 1, 1, 1, 0]),
            sources_to_check_for_pairwise=pairwise_source_info_both_classical),
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

class TestCommutations(unittest.TestCase):
    scenario = InflationProblem(dag={"s1": ["A", "B", "l1", "l2"],
                                     "s2": ["C", "D"],
                                     "s3": ["l2"],
                                     "l1": ["C", "D"],
                                     "l2": ["B", "E"]},
                                outcomes_per_party=[2, 2, 2, 2, 2],
                                settings_per_party=[2, 2, 2, 2, 2],
                                inflation_level_per_source=[2, 2, 2],
                                classical_sources=["s2"],
                                nonclassical_intermediate_latents=["l1"],
                                classical_intermediate_latents=["l2"])
    notcomm = scenario._default_notcomm
    meas    = scenario.measurements
    order   = scenario._lexorder
    
    def test_commute_itself(self):
        # All operators should commute with themselves.
        self.assertEqual(self.notcomm.diagonal().sum(),
                         0,
                         "Some operator does not commute with itself.")
    
    def test_same_nonclassical(self):
        # The operators corresponding to different measurements of a same party
        # fed by nonclassical sources do not commute. Take for example A.
        op1_init = np.where(np.all(self.order == self.meas[0][0,0,0], axis=1))[0]
        op1_end  = np.where(np.all(self.order == self.meas[0][0,0,-1], axis=1))[0]
        op2_init = np.where(np.all(self.order == self.meas[0][0,-1,0], axis=1))[0]
        op2_end  = np.where(np.all(self.order == self.meas[0][0,-1,-1], axis=1))[0]
        op1_init, op1_end = op1_init.item(), op1_end.item()
        op2_init, op2_end = op2_init.item(), op2_end.item()
        self.assertTrue(
            np.all(self.notcomm[op1_init:op1_end+1, op2_init:op2_end+1]),
            "Operators for different measurement of a same party commute.")
        
    def test_same_classical(self):
        # All operators corresponding to a same party fed by classical sources
        # commute. Take for example E.
        op1_init = np.where(np.all(self.order == self.meas[-1][0,0,0], axis=1))[0]
        op1_end  = np.where(np.all(self.order == self.meas[-1][-1,-1,-1], axis=1))[0]
        op1_init, op1_end = op1_init.item(), op1_end.item()
        self.assertEqual(
            np.sum(self.notcomm[op1_init:op1_end+1, op1_init:op1_end+1]),
            0,
            "Operators of a classical party do not commute.")

    def test_source_parties(self):
        # We restrict to the case of parties connected (or not) by a source.
        # Regardless of the nature of the sources, they should commute. This
        # applies to all pairs of parties, except BE (connected by a classical
        # latent, tretated later), and CD (connected by a quantum latent and a
        # classical source, so they only commute if the nonclassical copies do
        # not overlap, treated later)
        for pair in [[0, 1], [0, 2], [0, 3], [0, 4],
                     [1, 2], [1, 3], [2, 4], [3, 4]]:
            op1_init = np.where(np.all(self.order == self.meas[pair[0]][0,0,0], axis=1))[0]
            op1_end  = np.where(np.all(self.order == self.meas[pair[0]][-1,-1,-1], axis=1))[0]
            op2_init = np.where(np.all(self.order == self.meas[pair[1]][0,0,0], axis=1))[0]
            op2_end  = np.where(np.all(self.order == self.meas[pair[1]][-1,-1,-1], axis=1))[0]
            op1_init, op1_end = op1_init.item(), op1_end.item()
            op2_init, op2_end = op2_init.item(), op2_end.item()
            self.assertEqual(self.notcomm[op1_init:op1_end+1, op2_init:op2_end+1].sum(),
                             0,
                             "Operators for separate parties " + \
                             f"{chr(65+pair[0])} and {chr(65+pair[1])} do not commute.")
    
    def test_classical_nonclassical_sources(self):
        # The CD case. Here we only have commutation if the nonclassical copies
        # are disjoint
        op1_init = np.where(np.all(self.order == self.meas[2][0,0,0], axis=1))[0]
        op1_end  = np.where(np.all(self.order == self.meas[2][3,-1,-1], axis=1))[0]
        op2_init = np.where(np.all(self.order == self.meas[3][-4,0,0], axis=1))[0]
        op2_end  = np.where(np.all(self.order == self.meas[3][-1,-1,-1], axis=1))[0]
        op1_init, op1_end = op1_init.item(), op1_end.item()
        op2_init, op2_end = op2_init.item(), op2_end.item()
        self.assertEqual(self.notcomm[op1_init:op1_end+1, op2_init:op2_end+1].sum(),
                         0,
                         "Operators for separate parties with classical and " \
                         + " nonclassical parents do not commute when " \
                         + "measuring on different nonclassical copies.")

    def test_nonclassical_latent(self):
        # When two parties share a common nonclassical latent ancestor, their
        # operators over the exact same copies do not commute.
        op1_init = np.where(np.all(self.order == self.meas[2][0,0,0], axis=1))[0]
        op1_end  = np.where(np.all(self.order == self.meas[2][0,-1,-1], axis=1))[0]
        op2_init = np.where(np.all(self.order == self.meas[3][0,0,0], axis=1))[0]
        op2_end  = np.where(np.all(self.order == self.meas[3][0,-1,-1], axis=1))[0]
        op1_init, op1_end = op1_init.item(), op1_end.item()
        op2_init, op2_end = op2_init.item(), op2_end.item()
        self.assertTrue(
            np.all(self.notcomm[op1_init:op1_end+1, op2_init:op2_end+1]),
            "Operators for parties with a common nonclassical latent " + \
            "ancestor commute for same nonclassical copies.")

    def test_classical_latent(self):
        # When two parties share a common classical latent ancestor, their
        # operators should always commute. This is the case for BE.
        op1_init = np.where(np.all(self.order == self.meas[1][0,0,0], axis=1))[0]
        op1_end  = np.where(np.all(self.order == self.meas[1][-1,-1,-1], axis=1))[0]
        op2_init = np.where(np.all(self.order == self.meas[4][0,0,0], axis=1))[0]
        op2_end  = np.where(np.all(self.order == self.meas[4][-1,-1,-1], axis=1))[0]
        op1_init, op1_end = op1_init.item(), op1_end.item()
        op2_init, op2_end = op2_init.item(), op2_end.item()
        self.assertEqual(
            np.sum(self.notcomm[op1_init:op1_end+1, op2_init:op2_end+1]),
            0,
            "Operators for parties with a common classical latent ancestor " + \
            "do not commute.")