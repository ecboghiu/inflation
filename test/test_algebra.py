import unittest

from causalinflation.general_tools import *
from causalinflation.fast_npa import *


class TestNCAlgebra(unittest.TestCase):
    triangle = np.array([[0, 1, 1],  # Each row es a state, each column is the parties that are fed by a state
                         [1, 1, 0],
                         [1, 0, 1]])
    ins       = [1, 1, 1]
    outs      = [2, 2, 2]
    inflation = [2, 2, 2]
    names = ['A', 'B', 'C']

    meas, subs = generate_parties(triangle, ins, outs, inflation, noncommuting=True)

    def test_ordering_parties(self):
        triangle = np.array([[0, 1, 1],  # Each row es a state, each column is the parties that are fed by a state
                         [1, 1, 0],
                         [1, 0, 1]])
        ins       = [1, 1, 1]
        outs      = [2, 2, 2]
        inflation = [2, 2, 2]
        names = ['A', 'B', 'C']

        meas, subs = generate_parties(triangle, ins, outs, inflation, noncommuting=True)

        monomial_string = 'A_0_1_1_0_0*A_0_1_2_0_0*C_2_0_1_0_0*B_1_2_0_0_0'
        #result = to_name(canonicalize(np.array(to_numbers(monomial_string, names)), meas, subs, names), names)
        result = to_name(to_canonical(np.array(to_numbers(monomial_string, names))), names)
        correct = 'A_0_1_1_0_0*A_0_1_2_0_0*B_1_2_0_0_0*C_2_0_1_0_0'
        self.assertEqual(result, correct, "Problem with ordering parties")

    def test_commutation(self):
        names = ['A', 'B', 'C']
        monomial_string = 'A_2_1_0_0_0*A_1_2_0_0_0*B_1_0_1_0_0'
        result = to_name(to_canonical(np.array(to_numbers(monomial_string, names))), names)
        correct =         'A_1_2_0_0_0*A_2_1_0_0_0*B_1_0_1_0_0'
        self.assertEqual(result, correct, "Problem with bringing monomial to representative form")

    def test_ordering(self):
        # Check if mon_lessthan_mon correctly checks the order of two monomials
        mon1 = np.array([[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [2, 1, 1, 0, 0, 0]])
        mon2 = np.array([[1, 1, 1, 0, 0, 0], [1, 2, 1, 0, 0, 0], [2, 1, 1, 0, 0, 0]])
        result = mon_lessthan_mon(mon1, mon2)
        correct = True
        self.assertEqual(result, correct, "Problem with mon_lessthan_mon")
