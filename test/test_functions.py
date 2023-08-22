import unittest
import numpy as np

from sympy import Symbol

from inflation import InflationProblem, InflationSDP, InflationLP
from inflation.sdp.fast_npa import nb_remove_sandwich
from inflation.sdp.quantum_tools import to_symbol, \
                                        party_physical_monomials_via_cliques
from itertools import combinations, product, permutations



class TestFunctions(unittest.TestCase):
    def test_remove_sandwich(self):
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
        
class TestPhysicalMonomialGeneration(unittest.TestCase):
    def _old_party_physical_monomials(self, lp, party, max_monomial_length):
        """Generate all possible non-negative monomials for a given party composed
        of at most ``max_monomial_length`` operators.

        Parameters
        ----------
        hypergraph : numpy.ndarray
            Hypergraph of the scenario.
        inflevels : np.ndarray
            The number of copies of each source in the inflated scenario.
        party : int
            Party index. NOTE: starting from 0
        max_monomial_length : int
            The maximum number of operators in the monomial.
        settings_per_party : List[int]
            List containing the cardinality of the input/measurement setting
            of each party.
        outputs_per_party : List[int]
            List containing the cardinality of the output/measurement outcome
            of each party.
        lexorder : numpy.ndarray
            A matrix storing the lexicographic order of operators. If an operator
            has lexicographic rank `i`, then it is placed at the ``i``-th row of
            lexorder.

        Returns
        -------
        List[numpy.ndarray]
            An array containing all possible positive monomials of the given
            length.
        """
        from inflation.sdp.fast_npa import mon_lexsorted, apply_source_perm
        from inflation.utils import format_permutations
        
        hypergraph = np.asarray(lp.hypergraph)
        nr_sources = hypergraph.shape[0]
        nr_properties = 1 + nr_sources + 2
        relevant_sources = np.flatnonzero(hypergraph[:, party])
        relevant_inflevels = lp.inflation_levels[relevant_sources]

        assert max_monomial_length <= min(relevant_inflevels), \
            ("You cannot have a longer list of commuting operators" +
            " than the minimum inflation level of said part.")

        # The strategy is building an initial non-negative monomial and apply all
        # inflation symmetries
        initial_monomial = np.zeros(
            (max_monomial_length, nr_properties), dtype=np.uint8)
        if max_monomial_length == 0:
            return initial_monomial[np.newaxis]
        initial_monomial[:, 0] = 1 + party
        for mon_idx in range(max_monomial_length):
            initial_monomial[mon_idx, 1:-2] = hypergraph[:, party] * (1 + mon_idx)
        inflation_equivalents = {initial_monomial.tobytes(): initial_monomial}
        all_permutations_per_relevant_source = [
            format_permutations(list(permutations(range(inflevel))))
            for inflevel in relevant_inflevels.flat]
        for permutation in product(*all_permutations_per_relevant_source):
            permuted = initial_monomial.copy()
            for perm_idx, source in enumerate(relevant_sources.flat):
                permuted = mon_lexsorted(apply_source_perm(permuted,
                                                        source,
                                                        permutation[perm_idx]),
                                        lp._lexorder)
            inflation_equivalents[permuted.tobytes()] = permuted

        # Insert all combinations of inputs and outputs
        template_mon = np.stack(tuple(inflation_equivalents.values()))
        del inflation_equivalents
        nr_possible_in = lp.setting_cardinalities[party]
        nr_possible_out = lp.outcome_cardinalities[party] - 1 # We always use one less
        new_monomials = np.broadcast_to(
            template_mon,
            (nr_possible_in ** max_monomial_length,
            nr_possible_out ** max_monomial_length) + template_mon.shape).copy()
        del template_mon
        for i, input_slice in enumerate(product(range(nr_possible_in),
                                                repeat=max_monomial_length)):
            new_monomials[i, :, :, :, -2] = input_slice
            for o, output_slice in enumerate(product(range(nr_possible_out),
                                                    repeat=max_monomial_length)):
                new_monomials[i, o, :, :, -1] = output_slice
        return new_monomials.transpose(
            (2, 0, 1, 3, 4)
        ).reshape((-1, max_monomial_length, nr_properties))
            
    def test_physical_mon_gen_1party_no_copies(self):
        # Test for 1 party, no copies, nofanout
        scenario = InflationProblem({'r': ['A']}, (3,), (3,), (1, ))
        lp = InflationLP(scenario, nonfanout=True)

        bool2lexorder = np.arange(lp._lexorder.shape[0])
        set_predicted = {tuple(bool2lexorder[e]) 
                         for e in lp._raw_monomials_as_lexboolvecs}
        set_correct = [self._old_party_physical_monomials(lp, 0, i) 
                       for i in range(1, min(lp.inflation_levels) + 1)]
        set_correct = set([()] + [tuple(lp.mon_to_lexrepr(e)) 
                                  for c in set_correct for e in c])
        self.assertTrue(set_correct == set_predicted,
            "The physical monomials sets are not equal " + 
            "in 1 party nonfanout LP.")

    def test_physical_mon_gen_1party_3_copies(self):
        # Test for 1 party, 3 copies, nofanout
        scenario = InflationProblem({'r': ['A']}, (3,), (3,), (3, ))
        lp = InflationLP(scenario, nonfanout=True)

        bool2lexorder = np.arange(lp._lexorder.shape[0])
        set_predicted = {tuple(bool2lexorder[e]) 
                         for e in lp._raw_monomials_as_lexboolvecs}
        set_correct = [self._old_party_physical_monomials(lp, 0, i)
                       for i in range(1, min(lp.inflation_levels) + 1)]
        set_correct = set([()] + [tuple(lp.mon_to_lexrepr(e)) 
                                  for c in set_correct for e in c])
        self.assertTrue(set_correct == set_predicted, 
                       "The physical monomials sets are not equal " +
                       "in 1 party nonfanout LP.")
        
    def test_physical_mon_gen_1party_2_nc_sources(self):
        scenario = InflationProblem({'s': ['A'], 't': ['A']},
                                    (3,), (3,), (3, 2))
        lp = InflationLP(scenario, nonfanout=True)
        physical_monomials = scenario._generate_compatible_monomials_given_party(0)

        bool2lexorder = np.arange(lp._lexorder.shape[0])
        set_predicted = {tuple(bool2lexorder[e]) 
                         for e in physical_monomials}
        set_correct = [self._old_party_physical_monomials(lp, 0, i)
                       for i in range(1, min(lp.inflation_levels) + 1)]
        set_correct = set([()] + [tuple(lp.mon_to_lexrepr(e)) 
                                  for c in set_correct for e in c])
        self.assertTrue(set_correct == set_predicted, 
                       "The physical monomials sets are not equal " +
                       "in 1 party nonfanout LP.")
        
        self.assertTrue(len(physical_monomials) == 253, 
                        "Wrong number of physical monomials generated.")
        
    def test_physical_mon_gen_1party_2_hybrid_sources(self):
        scenario = InflationProblem({'s': ['A'], 't': ['A']},
                                    (3,), (2,), 
                                    (3, 2),
                                    classical_sources=['t'])
        physical_monomials = scenario._generate_compatible_monomials_given_party(0)
        self.assertTrue(len(physical_monomials) == 729, 
                        "Wrong number of physical monomials generated.")
        for boolmon in physical_monomials:
            _s_ = np.nonzero(boolmon)[0]
            _dim = _s_.size
            if _dim > 0:
                self.assertTrue(np.allclose(
                    scenario._compatible_measurements[np.ix_(_s_, _s_)],
                    np.ones((_dim,)*2)-np.eye(_dim)),
                    "Measurements that are supposed to be compatible are not.")

            