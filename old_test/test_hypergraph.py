import unittest
import numpy as np

from causalinflation.quantum.general_tools import is_knowable
from causalinflation import InflationProblem

class TestHypergraphUtils(unittest.TestCase):

    # Triangle scenario. The structure is
    #    A
    #  /  \
    # B - C
    # Each row is a state and columns are the parties that are fed by a state
    triangle = np.array([[1, 0, 1],
                         [1, 1, 0],
                         [0, 1, 1]])

    # Bowtie scenario. The structure is
    # A     D
    # | \ / |
    # |  C  |
    # | / \ |
    # B     E
    # Each row is a state and columns are the parties that are fed by a state
    bowtie = np.array([[1, 0, 1, 0, 0],
                       [1, 1, 0, 0, 0],
                       [0, 1, 1, 0, 0],
                       [0, 0, 1, 1, 0],
                       [0, 0, 1, 0, 1],
                       [0, 0, 0, 1, 1]])

    def test_knowable_complete(self):
        # A12B23C31 in the triangle scenario should be knowable
        A12B23C31 = np.array([[1, 2, 0, 1, None, None],
                              [2, 2, 3, 0, None, None],
                              [3, 0, 3, 1, None, None]])

        self.assertTrue(is_knowable(A12B23C31),
                        "A complete copy of the scenario is not knowable")

    def test_unknowable_complete(self):
        # A11B12C22 in the triangle scenario should not be knowable
        A11B12C22 = np.array([[1, 1, 0, 1, None, None],
                              [2, 1, 2, 0, None, None],
                              [3, 0, 2, 2, None, None]])

        self.assertFalse(is_knowable(A11B12C22),
                         "An open copy of the triangle is knowable")

    def test_knowable_partial(self):
        # A11 in the triangle should be knowable
        A11 = np.array([[1, 1, 0, 1, None, None]])

        self.assertTrue(is_knowable(A11),
                         "Knowable monomials of a subset of parties "
                         + "are unknowable")

    def test_unknowable_partial(self):
        # A11B11C1211 in the bowtie (with center on C) should not be knowable
        A11B11C1211 = np.array([[1, 1, 0, 1, 0, 0, 0, None, None],
                                [2, 1, 1, 0, 0, 0, 0, None, None],
                                [3, 0, 1, 2, 1, 1, 0, None, None]])

        self.assertFalse(is_knowable(A11B11C1211),
                         "Unknowable monomials of a subset of parties "
                         + "are knowable")

if __name__ == "__main__":
    unittest.main()
