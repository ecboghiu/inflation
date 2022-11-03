import unittest
import numpy as np

from inflation import InflationProblem


class TestFactorization(unittest.TestCase):
    def test_factorizable_NC(self):
        prob = InflationProblem(dag={"h1": ["v2", "v3"],
                                     "h2": ["v1", "v3"],
                                     "h3": ["v1", "v2"]},
                                outcomes_per_party=[2, 2, 2],
                                inflation_level_per_source=[6, 6, 3])
        # monomial = < A^011 * B^102 * A^033 * C^350 * C^140 * C^660 * C^450 >
        monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [2, 1, 0, 2, 0, 0],
                             [1, 0, 3, 3, 0, 0],
                             [3, 3, 5, 0, 0, 0],
                             [3, 1, 4, 0, 0, 0],
                             [3, 6, 6, 0, 0, 0],
                             [3, 4, 5, 0, 0, 0]])
        factorised = prob.factorize_monomial(monomial, canonical_order=True)
        correct    = np.array([np.array([[1, 0, 1, 1, 0, 0]]),
                               np.array([[1, 0, 3, 3, 0, 0]]),
                               np.array([[2, 1, 0, 2, 0, 0],
                                         [3, 1, 4, 0, 0, 0]]),
                               np.array([[3, 3, 5, 0, 0, 0],
                                         [3, 4, 5, 0, 0, 0]]),
                               np.array([[3, 6, 6, 0, 0, 0]])], dtype=object)

        self.assertEqual(len(correct), len(factorised),
                         "The factorization is not finding all factors.")
        for idx in range(len(correct)):
            self.assertTrue(np.allclose(correct[idx], factorised[idx]),
                            "The factors found are not in canonical form.")

    def test_higherorder(self):
        prob = InflationProblem(dag={"h1": ["v1", "v2", "v3"],
                                     "h2": ["v2", "v3", "v4"]},
                                outcomes_per_party=[2, 2, 2, 2],
                                settings_per_party=[2, 1, 2, 2],
                                inflation_level_per_source=[4, 4])
        monomial = np.array([[1, 1, 0, 1, 0],
                             [1, 1, 0, 0, 0],
                             [1, 2, 0, 1, 0],
                             [2, 2, 3, 0, 0],
                             [2, 1, 1, 0, 0],
                             [2, 1, 2, 0, 0],
                             [3, 3, 4, 0, 0],
                             [3, 1, 1, 0, 0],
                             [3, 4, 3, 1, 0],
                             [4, 0, 3, 1, 0],
                             [4, 0, 2, 1, 0],
                             [4, 0, 1, 1, 0]])
        factorised = prob.factorize_monomial(monomial)
        correct    = np.array([np.array([[1, 1, 0, 1, 0],
                                         [1, 1, 0, 0, 0],
                                         [2, 1, 1, 0, 0],
                                         [2, 1, 2, 0, 0],
                                         [3, 1, 1, 0, 0],
                                         [4, 0, 2, 1, 0],
                                         [4, 0, 1, 1, 0]]),
                               np.array([[3, 3, 4, 0, 0]]),
                               np.array([[3, 4, 3, 1, 0],
                                         [4, 0, 3, 1, 0],
                                         [1, 2, 0, 1, 0],
                                         [2, 2, 3, 0, 0]])], dtype=object)

        self.assertEqual(len(correct), len(factorised),
                         "The factorization with sources for n>2 parties is " +
                         "not working correctly.")

    def test_independent_parties(self):
        # In the bilocal scenario A - B - C, both A1C1 and A1C2 factorize
        # (because A and C are not causally connected). This should be taken
        # into account because A1C1 = A1C2 in the bilocal scenario
        # Input and output are set to 0 because they are irrelevant.
        prob = InflationProblem(dag={"h1": ["v1", "v2"],
                                     "h2": ["v2", "v3"]},
                                outcomes_per_party=[2, 2, 2],
                                inflation_level_per_source=[2, 2])
        A1C1 = np.array([[0, 1, 0, 0, 0],
                         [2, 0, 1, 0, 0]])
        A1C2 = np.array([[0, 1, 0, 0, 0],
                         [2, 0, 2, 0, 0]])

        self.assertEqual(len(prob.factorize_monomial(A1C1)),
                         len(prob.factorize_monomial(A1C2)),
                         "Causally independent parties are being treated "
                         + "different depending on the copy indices.")

    def test_unfactorizable(self):
        prob = InflationProblem(dag={"h1": ["v2", "v3"],
                                     "h2": ["v1", "v2"],
                                     "h3": ["v1", "v3"]},
                                outcomes_per_party=[2, 2, 2],
                                inflation_level_per_source=[2, 2, 2])
        # monomial = < A^011_00 * B^101_00 * C^120_00 >
        monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [2, 1, 1, 0, 0, 0],
                             [3, 1, 0, 2, 0, 0]])
        factorised = prob.factorize_monomial(monomial)

        self.assertEqual(monomial.tolist(), factorised[0].tolist(),
                         "Non-factorizable monomials are being factorized.")

    def test_unfactorizable_NC(self):
        # monomial = < A_00 * A_10 * A_00 >
        prob = InflationProblem(outcomes_per_party=[2],
                                settings_per_party=[2])
        monomial = np.array([[1, 1, 0, 0],
                             [1, 1, 1, 0],
                             [1, 1, 0, 0]])
        factorised = prob.factorize_monomial(monomial)

        self.assertEqual(monomial.tolist(), factorised[0].tolist(),
                         "Non-factorizable, non-commutative monomials are "
                         + "being reordered as if they were commutative.")
