import unittest
import numpy as np

from causalinflation.quantum.general_tools import to_numbers, to_representative
from causalinflation.quantum.fast_npa import to_name

class TestToRepr(unittest.TestCase):
    def test_to_representative(self):
        names = ['A', 'B']
        initial = 'A_1_1_0_0_0*A_2_2_0_0_0*B_2_0_1_0_0'
        correct = 'A_1_1_0_0_0*A_2_2_0_0_0*B_1_0_1_0_0'
        result  = to_name(to_representative(np.array(to_numbers(initial,
                                                                names)),
                                            np.array([2, 2]),
                                            False), names)
        self.assertEqual(result, correct,
                        "Problem with bringing monomial to representative form")

        initial = 'A_1_1_0_0_0*A_2_2_0_0_0*B_1_0_2_0_0*B_2_0_1_0_0'
        correct = 'A_1_1_0_0_0*A_2_2_0_0_0*B_1_0_1_0_0*B_2_0_2_0_0'
        result  = to_name(to_representative(np.array(to_numbers(initial,
                                                                names)),
                                            np.array([2, 2]),
                                            False), names)
        self.assertEqual(result, correct,
                        "Problem with bringing monomial to representative form")

        initial = 'A_3_1_0_0*A_2_1_0_0*A_3_1_0_0'
        correct = 'A_1_1_0_0*A_2_1_0_0*A_1_1_0_0'
        result  = to_name(to_representative(np.array(to_numbers(initial,
                                                                names)),
                                            np.array([2, 2]),
                                            False), names)
        self.assertEqual(result, correct,
                        "Problem with bringing monomial to representative form")

        correct = 'A_1_1_0_0*A_2_1_0_0'
        result  = to_name(to_representative(np.array(to_numbers(initial,
                                                                names)),
                                            np.array([2, 2]),
                                            True), names)
        self.assertEqual(result, correct,
             "Problem with bringing commuting monomials to representative form")
