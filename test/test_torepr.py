import unittest

from causalinflation.quantum.general_tools import *
from causalinflation.quantum.fast_npa import *

class TestToRepr(unittest.TestCase):
    def test_to_representative(self):
        names = ['A', 'B', 'C']
        monomial_string = 'A_1_1_0_0_0*A_2_2_0_0_0*B_2_0_1_0_0'
        result = to_name(to_representative(np.array(to_numbers(monomial_string, names)), np.array([2, 2, 2])), names)
        correct =         'A_1_1_0_0_0*A_2_2_0_0_0*B_1_0_1_0_0'
        self.assertEqual(result, correct, "Problem with bringing monomial to representative form")

        monomial_string = 'A_1_1_0_0_0*A_2_2_0_0_0*B_1_0_2_0_0*B_2_0_1_0_0'
        result = to_name(to_representative(np.array(to_numbers(monomial_string, names)), np.array([2, 2, 2])), names)
        correct = 'A_1_1_0_0_0*A_2_2_0_0_0*B_1_0_1_0_0*B_2_0_2_0_0'
        self.assertEqual(result, correct, "Problem with bringing monomial to representative form")
