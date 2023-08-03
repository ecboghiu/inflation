import unittest
import numpy as np

from sympy import Symbol

from inflation import InflationProblem, InflationSDP
from inflation.sdp.fast_npa import nb_remove_sandwich
from inflation.sdp.quantum_tools import to_symbol


class TestFunctions(unittest.TestCase):
    def test_sandwich(self):
        # <(A_111*A_121*A_111)*(A_332*A_312*A_342*A_312*A_332)*(B_011*B_012)>
        monomial = np.array([[1, 1, 1, 1, 0, 0],
                             [1, 1, 2, 1, 0, 0],
                             [1, 1, 1, 1, 0, 0],
                             [1, 3, 3, 2, 0, 0],
                             [1, 3, 5, 2, 0, 0],
                             [1, 3, 4, 2, 0, 0],
                             [1, 3, 5, 2, 0, 0],
                             [1, 3, 3, 2, 0, 0],
                             [2, 0, 1, 1, 0, 0],
                             [2, 0, 1, 2, 0, 0]])

        delayered = nb_remove_sandwich(monomial)
        correct = np.array([[1, 1, 2, 1, 0, 0],
                            [1, 3, 4, 2, 0, 0],
                            [2, 0, 1, 1, 0, 0],
                            [2, 0, 1, 2, 0, 0]])

        self.assertTrue(np.array_equal(delayered, correct),
                        "Removal of complex sandwiches is not working.")

    def test_sanitize(self):
        bellScenario = InflationProblem({"Lambda": ["A"]},
                                        outcomes_per_party=[3],
                                        settings_per_party=[2],
                                        inflation_level_per_source=[1])
        sdp = InflationSDP(bellScenario)
        sdp.generate_relaxation("npa1")
        monom = sdp.monomials[-1]
        self.assertEqual(monom, sdp._sanitise_monomial(monom),
                         f"Sanitization of {monom} as a CompoundMonomial is " +
                         f"giving {sdp._sanitise_monomial(monom)}.")
        mon1  = sdp.measurements[0][0][0][0]
        mon2  = sdp.measurements[0][0][0][1]
        mon3  = sdp.measurements[0][0][1][0]
        # Tests for symbols and combinations
        mon   = mon1
        truth = sdp.monomials[2]
        self.assertEqual(sdp._sanitise_monomial(mon), truth,
                         f"Sanitization of {mon} as a Symbol is giving " +
                         f"{sdp._sanitise_monomial(mon)} instead of {truth}.")
        mon = mon1**2
        truth = sdp.monomials[2]
        self.assertEqual(sdp._sanitise_monomial(mon), truth,
                         f"Sanitization of {mon} as a Power is giving " +
                         f"{sdp._sanitise_monomial(mon)} instead of {truth}.")
        mon   = mon1*mon2
        truth = sdp.Zero
        self.assertEqual(sdp._sanitise_monomial(mon), truth,
                         f"Sanitization of {mon} as a Mul is giving " +
                         f"{sdp._sanitise_monomial(mon)} instead of {truth}.")
        mon   = mon1*mon3
        truth = sdp.monomials[6]
        self.assertEqual(sdp._sanitise_monomial(mon), truth,
                         f"Sanitization of {mon} as a Mul is giving " +
                         f"{sdp._sanitise_monomial(mon)} instead of {truth}.")
        mon   = mon3*mon1
        truth = sdp.monomials[6]
        self.assertEqual(sdp._sanitise_monomial(mon), truth,
                         f"Sanitization of {mon} as a Mul is giving " +
                         f"{sdp._sanitise_monomial(mon)} instead of {truth}.")
        # Tests for array forms
        mon   = [[1, 1, 0, 0],
                 [1, 1, 1, 0]]
        truth = sdp.monomials[6]
        self.assertEqual(sdp._sanitise_monomial(mon), truth,
                         f"Sanitization of {mon} is giving " +
                         f"{sdp._sanitise_monomial(mon)} instead of {truth}.")
        mon   = [[1, 1, 1, 0],
                 [1, 1, 0, 0]]
        truth = sdp.monomials[6]
        self.assertEqual(sdp._sanitise_monomial(mon), truth,
                         f"Sanitization of {mon} is giving " +
                         f"{sdp._sanitise_monomial(mon)} instead of {truth}.")
        mon   = [[1, 1, 0, 0],
                 [1, 1, 0, 1]]
        truth = sdp.Zero
        self.assertEqual(sdp._sanitise_monomial(mon), truth,
                         f"Sanitization of {mon} is giving " +
                         f"{sdp._sanitise_monomial(mon)} instead of {truth}.")
        mon   = [[1, 1, 0, 0],
                 [1, 1, 0, 0]]
        truth = sdp.monomials[2]
        self.assertEqual(sdp._sanitise_monomial(mon), truth,
                         f"Sanitization of {mon} is giving " +
                         f"{sdp._sanitise_monomial(mon)} instead of {truth}.")
        # Tests for string forms
        mon   = "pA(0|0)"
        truth = sdp.monomials[2]
        self.assertEqual(sdp._sanitise_monomial(mon), truth,
                         f"Sanitization of {mon} is giving " +
                         f"{sdp._sanitise_monomial(mon)} instead of {truth}.")
        mon   = "<A_1_0_0 A_1_1_0>"
        truth = sdp.monomials[6]
        self.assertEqual(sdp._sanitise_monomial(mon), truth,
                         f"Sanitization of {mon} is giving " +
                         f"{sdp._sanitise_monomial(mon)} instead of {truth}.")
        mon   = "<A_1_1_0 A_1_0_0>"
        truth = sdp.monomials[6]
        self.assertEqual(sdp._sanitise_monomial(mon), truth,
                         f"Sanitization of {mon} is giving " +
                         f"{sdp._sanitise_monomial(mon)} instead of {truth}.")
        # Tests for number forms
        mon   = 0
        truth = sdp.Zero
        self.assertEqual(sdp._sanitise_monomial(mon), truth,
                         f"Sanitization of {mon} is giving " +
                         f"{sdp._sanitise_monomial(mon)} instead of {truth}.")
        mon   = 1
        truth = sdp.One
        self.assertEqual(sdp._sanitise_monomial(mon), truth,
                         f"Sanitization of {mon} is giving " +
                         f"{sdp._sanitise_monomial(mon)} instead of {truth}.")

    def test_to_symbol(self):
        truth = (Symbol("A_1_0_0", commutative=False)
                 * Symbol("B_1_1_0", commutative=False))

        self.assertEqual(to_symbol(np.array([[1, 1, 0, 0], [2, 1, 1, 0]]),
                                   ["A", "B"]),
                         truth,
                         "to_symbol is not working as expected.")

    def test_physical_monomial_generation(self):
        from inflation.sdp.quantum_tools import party_physical_monomials_via_cliques
        from itertools import combinations
        _lexorder = np.array(['a','b','c','d','e','f'], dtype=object)
        name2lexorder = {name: i for i, name in enumerate(_lexorder)}
        sets_of_notcommuting = [('a','b','c'), ('d','e'), ('f')]
        _notcomm = np.zeros([_lexorder.shape[0]]*2, dtype=bool)
        for clique in sets_of_notcommuting:
            for e1, e2 in combinations(clique, 2):
                _notcomm[name2lexorder[e1], name2lexorder[e2]] = True
                _notcomm[name2lexorder[e2], name2lexorder[e1]] = True
        result = party_physical_monomials_via_cliques(1, _lexorder, _notcomm)
        correct = np.array([[['a']],
                            [['b']], 
                            [['c']], 
                            [['d']],
                            [['e']],
                            [['f']]], dtype='<U1')
        self.assertTrue(np.array_equal(result, correct),
            "The physical monomials of length 1 are not generated correctly.")
        result = party_physical_monomials_via_cliques(2, _lexorder, _notcomm)
        correct = np.array([[['a'], ['d']],
                            [['a'], ['e']],
                            [['b'], ['d']],
                            [['b'], ['e']],
                            [['c'], ['d']],
                            [['c'], ['e']],
                            [['a'], ['f']],
                            [['b'], ['f']],
                            [['c'], ['f']],
                            [['d'], ['f']],
                            [['e'], ['f']]], dtype='<U1')
        self.assertTrue(np.array_equal(result, correct),
            "The physical monomials of length 2 are not generated correctly.")
        result = party_physical_monomials_via_cliques(3, _lexorder, _notcomm)
        correct = np.array([[['a'], ['d'], ['f']],
                            [['a'], ['e'], ['f']],
                            [['b'], ['d'], ['f']],
                            [['b'], ['e'], ['f']],
                            [['c'], ['d'], ['f']],
                            [['c'], ['e'], ['f']]], dtype='<U1')
        self.assertTrue(np.array_equal(result, correct),
            "The physical monomials of length 3 are not generated correctly.")
        
        # The following checks that the function raises the correct exception
        # when the monomial length is too large.
        with self.assertRaises(AssertionError): 
            party_physical_monomials_via_cliques(4, _lexorder, _notcomm)
