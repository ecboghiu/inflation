import unittest
import numpy as np

from causalinflation.quantum.general_tools import (is_knowable, is_physical,
                                                   to_numbers, to_representative
                                                   )
from causalinflation.quantum.fast_npa import (commutation_matrix, to_canonical,
                                              to_name)

class TestKnowable(unittest.TestCase):
    # Triangle scenario. The structure is
    #    A
    #  /  \
    # B - C
    triangle = np.array([[1, 0, 1],
                         [1, 1, 0],
                         [0, 1, 1]])

    # Bowtie scenario. The structure is
    # A     D
    # | \ / |
    # |  C  |
    # | / \ |
    # B     E
    bowtie = np.array([[1, 0, 1, 0, 0],
                       [1, 1, 0, 0, 0],
                       [0, 1, 1, 0, 0],
                       [0, 0, 1, 1, 0],
                       [0, 0, 1, 0, 1],
                       [0, 0, 0, 1, 1]])

    def test_knowable_complete(self):
        A12B23C31 = np.array([[1, 2, 0, 1, None, None],
                              [2, 2, 3, 0, None, None],
                              [3, 0, 3, 1, None, None]])

        self.assertTrue(is_knowable(A12B23C31),
                        "Monomials composed of a complete copy of the scenario "
                        + "are not identified as knowable.")

    def test_knowable_partial(self):
        # A11 in the triangle should be knowable
        A11 = np.array([[1, 1, 0, 1, None, None]])

        self.assertTrue(is_knowable(A11),
                        "Knowable monomials composed of a subset of parties "
                         + "are not identified as knowable.")

    def test_unknowable_complete(self):
        # A11B12C22 in the triangle scenario should not be knowable
        A11B12C22 = np.array([[1, 1, 0, 1, None, None],
                              [2, 1, 2, 0, None, None],
                              [3, 0, 2, 2, None, None]])

        self.assertFalse(is_knowable(A11B12C22),
                         "Monomials composed of an open copy of the triangle " +
                         "scenario are not identified as unknowable.")

    def test_unknowable_partial(self):
        # A11B11C1211 in the bowtie should not be knowable
        A11B11C1211 = np.array([[1, 1, 0, 1, 0, 0, 0, None, None],
                                [2, 1, 1, 0, 0, 0, 0, None, None],
                                [3, 0, 1, 2, 1, 1, 0, None, None]])

        self.assertFalse(is_knowable(A11B11C1211),
                         "Unknowable monomials composed of a subset of parties "
                         + "are not identified as unknowable.")


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
                         "Physical monomials with multiple copies of a party " +
                         "are not identified as such.")

    def test_two_copies_unphysical(self):
        monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [1, 0, 2, 1, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [3, 1, 2, 0, 0, 0]])
        self.assertEqual(is_physical(monomial), False,
                         "Unphysical monomials are not identified as " +
                         "unphysical.")


class TestToCanonical(unittest.TestCase):
    names = ('A', 'B', 'C')
    lexorder = np.array([[1, 1, 1, 0, 0, 0],
                         [1, 1, 2, 0, 0, 0],
                         [1, 2, 1, 0, 0, 0],
                         [1, 2, 2, 0, 0, 0],
                         [1, 3, 1, 0, 0, 0],
                         [1, 3, 2, 0, 0, 0],
                         [2, 1, 0, 1, 0, 0],
                         [2, 1, 0, 2, 0, 0],
                         [2, 2, 0, 1, 0, 0],
                         [2, 2, 0, 2, 0, 0],
                         [3, 0, 1, 1, 0, 0],
                         [3, 0, 1, 2, 0, 0],
                         [3, 0, 2, 1, 0, 0],
                         [3, 0, 2, 2, 0, 0]])
    notcomm = commutation_matrix(lexorder)

    def test_commutation(self):
        monomial_string = 'A_2_1_0_0_0*A_1_2_0_0_0*B_1_0_1_0_0'
        result  = to_name(
                      to_canonical(
                          np.array(to_numbers(monomial_string, self.names)),
                      self.notcomm,
                      self.lexorder),
                      self.names)
        correct = 'A_1_2_0_0_0*A_2_1_0_0_0*B_1_0_1_0_0'
        self.assertEqual(result, correct,
                         "to_canonical has problems with bringing monomials " +
                         "to representative form.")

    def test_ordering_parties(self):
        monomial_string = 'A_1_1_0_0_0*A_1_2_0_0_0*C_0_2_1_0_0*B_1_0_2_0_0'
        result  = to_name(
                      to_canonical(
                          np.array(to_numbers(monomial_string, self.names)),
                      self.notcomm,
                      self.lexorder),
                      self.names)
        correct = 'A_1_1_0_0_0*A_1_2_0_0_0*B_1_0_2_0_0*C_0_2_1_0_0'
        self.assertEqual(result, correct,
                         "to_canonical fails to order parties correctly.")


class TestToRepr(unittest.TestCase):
    lexorder = np.array([[1, 1, 1, 0, 0, 0],
                         [1, 1, 2, 0, 0, 0],
                         [1, 2, 1, 0, 0, 0],
                         [1, 2, 2, 0, 0, 0],
                         [1, 3, 1, 0, 0, 0],
                         [1, 3, 2, 0, 0, 0],
                         [2, 1, 0, 1, 0, 0],
                         [2, 1, 0, 2, 0, 0],
                         [2, 2, 0, 1, 0, 0],
                         [2, 2, 0, 2, 0, 0],
                         [3, 0, 1, 1, 0, 0],
                         [3, 0, 1, 2, 0, 0],
                         [3, 0, 2, 1, 0, 0],
                         [3, 0, 2, 2, 0, 0]], dtype=np.uint8)
    notcomm = commutation_matrix(lexorder)

    names = ('A', 'B', 'C')
    def test_commuting(self):
        initial = 'A_3_1_0_0_0*A_2_1_0_0_0*A_3_1_0_0_0'
        correct = 'A_1_1_0_0_0*A_2_1_0_0_0'
        initial_array = np.array(to_numbers(initial, self.names), dtype=np.uint8)
        result = to_name(to_representative(initial_array,
                                            np.array([2, 1, 1]),
                                            self.notcomm,
                                            self.lexorder,
                                            True), self.names)
        self.assertEqual(result, correct,
                         "Applying commutations for representative form fails.")

    def test_jump_sources(self):
        initial = 'A_3_1_0_0_0*A_2_1_0_0_0*A_3_1_0_0_0'
        correct = 'A_1_1_0_0_0*A_2_1_0_0_0*A_1_1_0_0_0'
        result  = to_name(to_representative(np.array(to_numbers(initial,
                                                    self.names)),
                                            np.array([2, 1, 1]),
                                            self.notcomm,
                                            self.lexorder,
                                            False), self.names)
        self.assertEqual(result, correct,
             "Skipping inflation indices when computing representatives fails.")

    def test_swap_all_sources(self):
        initial = 'A_1_1_0_0_0*A_2_2_0_0_0*B_2_0_1_0_0'
        correct = 'A_1_1_0_0_0*A_2_2_0_0_0*B_1_0_1_0_0'
        result  = to_name(to_representative(np.array(to_numbers(initial,
                                                                self.names)),
                                            np.array([2, 2, 2]),
                                            self.notcomm,
                                            self.lexorder,
                                            False), self.names)
        self.assertEqual(result, correct,
          "Swapping all sources and applying factorization commutations fails.")

    def test_swap_single_source(self):
        initial = 'A_1_1_0_0_0*A_2_2_0_0_0*B_1_0_2_0_0*B_2_0_1_0_0'
        correct = 'A_1_1_0_0_0*A_2_2_0_0_0*B_1_0_1_0_0*B_2_0_2_0_0'
        result  = to_name(to_representative(np.array(to_numbers(initial,
                                                                self.names)),
                                            np.array([2, 2, 2]),
                                            self.notcomm,
                                            self.lexorder,
                                            False), self.names)
        self.assertEqual(result, correct,
                        "Swapping a single source is failing.")
