import unittest
import numpy as np
import warnings
from causalinflation.quantum.general_tools import remove_sandwich, is_physical
class Monomial(object):
    def __init__(self, array2d, **kwargs):
        self.physical_q = is_physical(array2d, **kwargs)


class QuantumInflationPhysicalMoments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore", category=DeprecationWarning)

    def test_is_physical(self):
        # Single operator per party but not known
        monomial = Monomial([[1, 0, 1, 1, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [3, 1, 2, 0, 0, 0]], sandwich_positivity=False)
        self.assertEqual(monomial.physical_q, True, "Problem determining if physical or not.")

        # Single operator per party but not known
        monomial = Monomial([[1, 0, 1, 1, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [3, 2, 1, 0, 0, 0]])
        self.assertEqual(monomial.physical_q, True, "Problem determining if physical or not.")

        monomial = Monomial([[1, 0, 1, 1, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [3, 1, 1, 0, 0, 0]])
        self.assertEqual(monomial.physical_q, True, "Problem determining if physical or not.")

        # Two non-commuting As or Bs or Cs
        monomial = Monomial([[1, 0, 1, 1, 0, 0],
                             [1, 0, 2, 1, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [3, 1, 2, 0, 0, 0]])
        self.assertEqual(monomial.physical_q, False, "Problem determining if physical or not.")

        monomial = Monomial([[1, 0, 1, 1, 0, 0],
                             [1, 0, 2, 2, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [3, 1, 2, 0, 0, 0]])
        self.assertEqual(monomial.physical_q, True, "Problem determining if physical or not.")

        monomial = Monomial([[1, 0, 1, 1, 0, 0],
                             [1, 0, 2, 2, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [3, 2, 1, 0, 0, 0]])
        self.assertEqual(monomial.physical_q, True, "Problem determining if physical or not.")

        monomial = Monomial([[1, 0, 1, 1, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [2, 2, 0, 2, 0, 0],
                             [3, 1, 2, 0, 0, 0]])
        self.assertEqual(monomial.physical_q, True, "Problem determining if physical or not.")

        monomial = Monomial([[1, 0, 1, 1, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [2, 2, 0, 2, 0, 0],
                             [3, 2, 1, 0, 0, 0]])
        self.assertEqual(monomial.physical_q, True, "Problem determining if physical or not.")

        monomial = Monomial([[1, 0, 1, 1, 0, 0],
                             [2, 1, 0, 2, 0, 0],
                             [3, 1, 1, 0, 0, 0],
                             [3, 2, 2, 0, 0, 0]])
        self.assertEqual(monomial.physical_q, True, "Problem determining if physical or not.")

        # Two operators per party, commuting within party
        monomial = Monomial([[1, 0, 1, 1, 0, 0],
                             [1, 0, 2, 2, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [2, 2, 0, 2, 0, 0],
                             [3, 1, 2, 0, 0, 0],
                             [3, 2, 1, 0, 0, 0]])
        self.assertEqual(monomial.physical_q, True, "Problem determining if physical or not.")

        # Variations of previous
        monomial = Monomial([[1, 0, 1, 1, 0, 0],
                             [1, 0, 2, 2, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [2, 2, 0, 2, 0, 0],
                             [3, 1, 2, 0, 0, 0],
                             [3, 2, 1, 0, 0, 0]])
        self.assertEqual(monomial.physical_q, True, "Problem determining if physical or not.")

        # Add another party, but still physical
        # < A11*A22*B11*B22*C12*C21*D(...) >
        monomial = Monomial([[1, 0, 1, 1, 0, 0],
                             [1, 0, 2, 2, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [2, 2, 0, 2, 0, 0],
                             [3, 1, 2, 0, 0, 0],
                             [3, 2, 1, 0, 0, 0],
                             [4, 2, 4, 4, 0, 0]])
        self.assertEqual(monomial.physical_q, True, "Problem determining if physical or not.")

        # Only 2 parties, not physical
        monomial = Monomial([[1, 0, 1, 1, 0, 0],
                             [1, 0, 1, 2, 0, 0],
                             [2, 1, 0, 1, 0, 0],
                             [2, 2, 0, 2, 0, 0]])
        self.assertEqual(monomial.physical_q, False, "Problem determining if physical or not.")

    def test_sandwich(self):

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
