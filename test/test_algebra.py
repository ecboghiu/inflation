import unittest
import warnings
import numpy as np

from causalinflation.quantum.general_tools import (to_name, to_canonical,
                                                   to_numbers)
from causalinflation.quantum.fast_npa import mon_lessthan_mon



class TestNCAlgebra(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore", category=DeprecationWarning)

    names = ['A', 'B', 'C']

    def test_ordering_parties(self):
        monomial_string = 'A_0_1_1_0_0*A_0_1_2_0_0*C_2_0_1_0_0*B_1_2_0_0_0'
        result  = to_name(
                      to_canonical(
                          np.array(to_numbers(monomial_string, self.names))),
                      self.names)
        correct = 'A_0_1_1_0_0*A_0_1_2_0_0*B_1_2_0_0_0*C_2_0_1_0_0'
        self.assertEqual(result, correct, "to_canonical fails to order parties")

    def test_commutation(self):
        monomial_string = 'A_2_1_0_0_0*A_1_2_0_0_0*B_1_0_1_0_0'
        result  = to_name(
                      to_canonical(
                          np.array(to_numbers(monomial_string, self.names))),
                      self.names)
        correct = 'A_1_2_0_0_0*A_2_1_0_0_0*B_1_0_1_0_0'
        self.assertEqual(result, correct,
                         "to_canonical has problems with bringing monomials " +
                         "to representative form")

    def test_ordering(self):
        # A_1_1_0_0_0*A_1_1_0_0_0*B_1_1_0_0_0
        mon1 = np.array([[1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [2, 1, 1, 0, 0, 0]])
        # A_1_1_0_0_0*A_2_1_0_0_0*B_1_1_0_0_0
        mon2 = np.array([[1, 1, 1, 0, 0, 0],
                         [1, 2, 1, 0, 0, 0],
                         [2, 1, 1, 0, 0, 0]])
        result = mon_lessthan_mon(mon1, mon2)
        correct = True
        self.assertEqual(result, correct,
                         "mon_lessthan_mon is not finding proper ordering")
