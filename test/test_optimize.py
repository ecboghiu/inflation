import unittest
import numpy as np
import warnings

from sympy import Symbol

from inflation import InflationProblem, InflationSDP, max_within_feasible


class TestOptimize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)

    def PRbox(v):
        p = np.zeros((2, 2, 2, 2), dtype=object)
        for a, b, x, y in np.ndindex((2, 2, 2, 2)):
            p[a, b, x, y] = 1/4 * (1 + v*(-1)**(a + b + x * y))
        return p

    precision = 1e-5
    bellScenario = InflationProblem({"Lambda": ["A", "B"]},
                                    outcomes_per_party=[2, 2],
                                    settings_per_party=[2, 2],
                                    inflation_level_per_source=[1])
    sdp = InflationSDP(bellScenario)
    sdp.generate_relaxation("npa1")
    sdp.set_distribution(PRbox(Symbol("v")))
    symbolic_values = sdp.known_moments

    def test_bisect(self):
        v_crit = max_within_feasible(self.sdp,
                                     self.symbolic_values,
                                     "bisection",
                                     precision=self.precision)
        self.assertTrue(np.isclose(v_crit, 1/np.sqrt(2), self.precision),
                        "Bisection of the quantum critical visibility for the "
                        + "PR box is not achieving 1/sqrt(2).")

    def test_dual(self):
        v_crit = max_within_feasible(self.sdp, self.symbolic_values, "dual",
                                     precision=self.precision)
        self.assertTrue(np.isclose(v_crit, 1/np.sqrt(2), self.precision),
                        "Dual optimization of the quantum critical visibility "
                        + "for the PR box is not achieving 1/sqrt(2).")
