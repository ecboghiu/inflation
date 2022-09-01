import unittest
import numpy as np
import warnings
from causalinflation.quantum.general_tools import to_numbers, to_representative
from causalinflation.quantum.fast_npa import to_name

class TestToRepr(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore", category=DeprecationWarning)

    def test_to_representative(self):
        from causalinflation.quantum.fast_npa import nb_commuting
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
        notcomm = np.zeros((lexorder.shape[0], lexorder.shape[0]), dtype=int)
        for i in range(lexorder.shape[0]):
            for j in range(i+1, lexorder.shape[0]):
                notcomm[i, j] = int(not nb_commuting(lexorder[i],
                                                        lexorder[j]))
        notcomm = notcomm + notcomm.T

        
        names = ['A', 'B']
        initial = 'A_1_1_0_0_0*A_2_2_0_0_0*B_2_0_1_0_0'
        correct = 'A_1_1_0_0_0*A_2_2_0_0_0*B_1_0_1_0_0'
        result  = to_name(to_representative(np.array(to_numbers(initial,
                                                                names)),
                                            np.array([2, 2, 2]),
                                            notcomm,
                                            lexorder,
                                            False), names)
        self.assertEqual(result, correct,
                        "Problem with bringing monomial to representative form 1")

        initial = 'A_1_1_0_0_0*A_2_2_0_0_0*B_1_0_2_0_0*B_2_0_1_0_0'
        correct = 'A_1_1_0_0_0*A_2_2_0_0_0*B_1_0_1_0_0*B_2_0_2_0_0'
        result  = to_name(to_representative(np.array(to_numbers(initial,
                                                                names)),
                                            np.array([2, 2, 2]),
                                            notcomm,
                                            lexorder,
                                            False), names)
        self.assertEqual(result, correct,
                        "Problem with bringing monomial to representative form 2")

        initial = 'A_3_1_0_0_0*A_2_1_0_0_0*A_3_1_0_0_0'
        correct = 'A_1_1_0_0_0*A_2_1_0_0_0*A_1_1_0_0_0'
        result  = to_name(to_representative(np.array(to_numbers(initial,
                                                                names)),
                                            np.array([2, 2, 2]),
                                            notcomm,
                                            lexorder,
                                            False), names)
        self.assertEqual(result, correct,
                        "Problem with bringing monomial to representative form 3")

        correct = 'A_1_1_0_0_0*A_2_1_0_0_0'
        result  = to_name(to_representative(np.array(to_numbers(initial,
                                                                names)),
                                            np.array([2, 2, 2]),
                                            notcomm,
                                            lexorder,
                                            True), names)
        self.assertEqual(result, correct,
                "Problem with bringing commuting monomials to representative form 4")
