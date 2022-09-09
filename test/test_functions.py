import unittest
import numpy as np

from causalinflation.quantum.fast_npa import mon_lessthan_mon
from causalinflation.quantum.general_tools import remove_sandwich

class TestFunctions(unittest.TestCase):
    def test_ordering(self):
        lexorder =  np.array([[1, 1, 1, 0, 0, 0],
                              [1, 2, 1, 0, 0, 0],
                              [2, 1, 1, 0, 0, 0]])
        # A_1_1_0_0_0*A_1_1_0_0_0*B_1_1_0_0_0
        mon1 = np.array([[1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [2, 1, 1, 0, 0, 0]])
        # A_1_1_0_0_0*A_2_1_0_0_0*B_1_1_0_0_0
        mon2 = np.array([[1, 1, 1, 0, 0, 0],
                         [1, 2, 1, 0, 0, 0],
                         [2, 1, 1, 0, 0, 0]])
        result = mon_lessthan_mon(mon1, mon2, lexorder)
        correct = True
        self.assertEqual(result, correct,
                         "mon_lessthan_mon is not finding proper ordering")

    def test_sandwich(self):
        # <(A_0111*A_0121*A_0111)*(A_332*A_0342*A_332)*(B_0011*B_0012)>
        monomial = np.array([[1, 0, 1, 1, 1, 0, 0],
                             [1, 0, 1, 2, 1, 0, 0],
                             [1, 0, 1, 1, 1, 0, 0],
                             [1, 0, 3, 3, 2, 0, 0],
                             [1, 0, 3, 4, 2, 0, 0],
                             [1, 0, 3, 3, 2, 0, 0],
                             [2, 0, 0, 1, 1, 0, 0],
                             [2, 0, 0, 1, 2, 0, 0]])

        delayered = remove_sandwich(monomial)
        correct = np.array([[1, 0, 1, 2, 1, 0, 0],
                            [1, 0, 3, 4, 2, 0, 0],
                            [2, 0, 0, 1, 1, 0, 0],
                            [2, 0, 0, 1, 2, 0, 0]])

        self.assertTrue(np.array_equal(delayered, correct),
                        "Removal of complex sandwiches is not working.")
