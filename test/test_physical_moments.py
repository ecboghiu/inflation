import unittest

from quantuminflation.general_tools import *

class QuantumInflationPhysicalMoments(unittest.TestCase):

    def test_is_physical(self):
        # Single operator per party but not known
        monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [3, 1, 2, 0, 0, 0]])
        self.assertEqual(is_physical(monomial), True, "Problem determining if physical or not.")

        # Single operator per party but not known
        monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [3, 2, 1, 0, 0, 0]])
        self.assertEqual(is_physical(monomial), True, "Problem determining if physical or not.")

        monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [3, 1, 1, 0, 0, 0]])
        self.assertEqual(is_physical(monomial), True, "Problem determining if physical or not.")

        # Two non-commuting As or Bs or Cs
        monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [1, 0, 2, 1, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [3, 1, 2, 0, 0, 0]])
        self.assertEqual(is_physical(monomial), False, "Problem determining if physical or not.")

        monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [1, 0, 2, 2, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [3, 1, 2, 0, 0, 0]])
        self.assertEqual(is_physical(monomial), True, "Problem determining if physical or not.")

        monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [1, 0, 2, 2, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [3, 2, 1, 0, 0, 0]])
        self.assertEqual(is_physical(monomial), True, "Problem determining if physical or not.")

        monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [2, 2, 0, 2, 0, 0],
                             [3, 1, 2, 0, 0, 0]])
        self.assertEqual(is_physical(monomial), True, "Problem determining if physical or not.")

        monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [2, 2, 0, 2, 0, 0],
                             [3, 2, 1, 0, 0, 0]])
        self.assertEqual(is_physical(monomial), True, "Problem determining if physical or not.")

        monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [2, 1, 0, 2, 0, 0],
                             [3, 1, 1, 0, 0, 0],
                             [3, 2, 2, 0, 0, 0]])
        self.assertEqual(is_physical(monomial), True, "Problem determining if physical or not.")

        # Two operators per party, commuting within party
        monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [1, 0, 2, 2, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [2, 2, 0, 2, 0, 0],
                             [3, 1, 2, 0, 0, 0],
                             [3, 2, 1, 0, 0, 0]])
        self.assertEqual(is_physical(monomial), True, "Problem determining if physical or not.")
        
        # Variations of previous
        monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [1, 0, 2, 2, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [2, 2, 0, 2, 0, 0],
                             [3, 1, 2, 0, 0, 0],
                             [3, 2, 1, 0, 0, 0]])
        self.assertEqual(is_physical(monomial), True, "Problem determining if physical or not.")

        # Add another party, but still physical
        # < A11*A22*B11*B22*C12*C21*D(...) >
        monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [1, 0, 2, 2, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [2, 2, 0, 2, 0, 0],
                             [3, 1, 2, 0, 0, 0],
                             [3, 2, 1, 0, 0, 0],
                             [4, 2, 4, 4, 0, 0]])
        self.assertEqual(is_physical(monomial), True, "Problem determining if physical or not.")

        # Only 2 parties, not physical
        monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [1, 0, 1, 2, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [2, 2, 0, 2, 0, 0]])
        self.assertEqual(is_physical(monomial), False, "Problem determining if physical or not.")

    def test_sandwhich(self):

        # <A021*A022*A021*B101*B202*C120*C210>  Hexagon structure
        monomial = np.array([[1, 0, 2, 1, 0, 0],
                             [1, 0, 2, 2, 0, 0],
                             [1, 0, 2, 1, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [2, 2, 0, 2, 0, 0],
                             [3, 1, 2, 0, 0, 0],
                             [3, 2, 1, 0, 0, 0]])
        delayered = remove_sandwich(monomial)
        correct =  np.array([[1, 0, 2, 2, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [2, 2, 0, 2, 0, 0],
                             [3, 1, 2, 0, 0, 0],
                             [3, 2, 1, 0, 0, 0]])
        self.assertTrue(np.array_equal(delayered, correct), "Doesn't recognize a sandwich-type moment.")

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

        self.assertTrue(np.array_equal(delayered, correct), "Doesn't recognize a sandwich-type moment.")
