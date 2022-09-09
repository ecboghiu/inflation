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
    lexorder =  np.array([[1, 0, 1, 1, 0, 0],
                          [1, 0, 1, 2, 0, 0],
                          [1, 0, 2, 1, 0, 0],
                          [1, 0, 2, 2, 0, 0],
                          [2, 1, 1, 0, 0, 0],
                          [2, 1, 2, 0, 0, 0],
                          [2, 2, 1, 0, 0, 0],
                          [2, 2, 2, 0, 0, 0],
                          [3, 1, 0, 1, 0, 0],
                          [3, 1, 0, 2, 0, 0],
                          [3, 2, 0, 1, 0, 0],
                          [3, 2, 0, 2, 0, 0]])
    
    notcomm = np.zeros((lexorder.shape[0], lexorder.shape[0]), dtype=int)
    notcomm[0, 1] = 1
    notcomm[0, 2] = 1
    notcomm[1, 3] = 1
    notcomm[2, 3] = 1
    notcomm[4, 5] = 1
    notcomm[4, 6] = 1
    notcomm[5, 7] = 1
    notcomm[5, 7] = 1
    notcomm = notcomm + notcomm.T

    def test_ordering_parties(self):
        monomial_string = 'A_0_1_1_0_0*A_0_1_2_0_0*C_2_0_1_0_0*B_1_2_0_0_0'
        result  = to_name(
                      to_canonical(
                          np.array(to_numbers(monomial_string, self.names)),
                          self.notcomm, self.lexorder),
                      self.names)
        correct = 'A_0_1_1_0_0*A_0_1_2_0_0*B_1_2_0_0_0*C_2_0_1_0_0'
        self.assertEqual(result, correct, "to_canonical fails to order parties")

    def test_commutation(self):
        monomial_string = 'B_1_1_0_0_0*A_0_2_1_0_0*A_0_1_2_0_0'
        result  = to_name(
                      to_canonical(
                          np.array(to_numbers(monomial_string, self.names)),
                          self.notcomm, self.lexorder),
                      self.names)
        
        correct = 'A_0_1_2_0_0*A_0_2_1_0_0*B_1_1_0_0_0'
        self.assertEqual(result, correct,
                         "to_canonical has problems with bringing monomials " +
                         "to representative form")
        
        # A32*A12*A21=A21*A32*A12
        from causalinflation.quantum.fast_npa import nb_mon_to_lexrepr
        lexorder = np.array([[1, 1, 1, 0, 0, 0],
                         [1, 1, 2, 0, 0, 0],
                         [1, 2, 1, 0, 0, 0],
                         [1, 2, 2, 0, 0, 0],
                         [1, 3, 2, 0, 0, 0],
                         [2, 1, 0, 1, 0, 0],
                         [2, 1, 0, 2, 0, 0],
                         [2, 2, 0, 1, 0, 0],
                         [2, 2, 0, 2, 0, 0],
                         [3, 0, 1, 1, 0, 0],
                         [3, 0, 1, 2, 0, 0],
                         [3, 0, 2, 1, 0, 0],
                         [3, 0, 2, 2, 0, 0]]) 
    
        notcomm = np.zeros((lexorder.shape[0], lexorder.shape[0]), dtype=int)
        notcomm[0, 1] = 1; notcomm[0, 2] = 1;
        notcomm[1, 3] = 1
        notcomm[1, 4] = 1
        notcomm[2, 3] = 1
        notcomm[3, 4] = 1
        notcomm[5, 6] = 1; notcomm[4, 6] = 1;
        notcomm[6, 8] = 1
        notcomm[7, 8] = 1
        notcomm[8, 10] = 1; notcomm[8, 10] = 1;
        notcomm[10, 12] = 1
        notcomm[11, 12] = 1
        notcomm = notcomm + notcomm.T
        
        # A32*A12*A21
        mon = np.array([[1, 3, 2, 0, 0, 0], [1, 1, 2, 0, 0, 0], [1, 2, 1, 0, 0, 0]])
        mon2 = to_canonical(mon, notcomm, lexorder)
        correct = np.array([[1, 2, 1, 0, 0, 0],[1, 3, 2, 0, 0, 0], [1, 1, 2, 0, 0, 0]])
        assert np.array_equal(mon2, correct), "to_canonical fails for A32*A12*A21=A21*A32*A12 "

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
