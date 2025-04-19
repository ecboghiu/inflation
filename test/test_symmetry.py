import unittest
import warnings
import numpy as np

from inflation import InflationProblem, InflationLP
from inflation.symmetry_utils import (discover_distribution_symmetries,
                                      group_elements_from_generators)


class TestSymmetry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore", category=UserWarning)

    PR_box = np.zeros((2, 2, 2, 2))
    for a,b,x,y in np.ndindex(*PR_box.shape):
        if np.bitwise_xor(a,b) == np.bitwise_and(x,y):
            PR_box[a,b,x,y] = 0.5

    GHZ = np.zeros((2, 2, 2, 1, 1, 1))
    GHZ[0, 0, 0, 0, 0, 0] = 1/2
    GHZ[1, 1, 1, 0, 0, 0] = 1/2

    bellScenario = InflationProblem({"Lambda": ["A", "B"]},
                                    outcomes_per_party=[2, 2],
                                    settings_per_party=[2, 2],
                                    inflation_level_per_source=[1],
                                    classical_sources='all')

    triangle = InflationProblem({"Lambda": ["A", "B"],
                                 "Mu": ["B", "C"],
                                 "Sigma": ["C", "A"]},
                                outcomes_per_party=[2, 2, 2],
                                inflation_level_per_source=[2, 1, 1])

    PRbox_symmetries = discover_distribution_symmetries(PR_box,
                                                        bellScenario)

    def test_discover(self):
        # Order: (a=0,x=0), (a=1,x=0), (a=0,x=1), (a=1,x=1),
        #        (b=0,y=0), (b=1,y=0), (b=0,y=1), (b=1,y=1)
        # There's five symmetries: flip parties, flip X, flip Y, and flip the
        # outcomes of each. Out of all these, the only valid ones are those that
        # do not change a+b+xy mod 2
        # Identity
        symmetries = [[0, 1, 2, 3, 4, 5, 6, 7]]
        # Flip outcomes in x=1, and flip y
        symmetries += [[0, 1, 3, 2, 6, 7, 4, 5]]
        # Flip outcomes in x=0, flip b and y
        symmetries += [[1, 0, 2, 3, 7, 6, 5, 4]]
        # Flip a and b
        symmetries += [[1, 0, 3, 2, 5, 4, 7, 6]]
        # Flip x, flip b outcomes in y=1
        symmetries += [[2, 3, 0, 1, 4, 5, 7, 6]]
        # Flip x, flip y, flip a in x=1, flip b in y=0
        symmetries += [[2, 3, 1, 0, 7, 6, 4, 5]]
        # Flip x, flip y, flip a in x=0, flip b in y=1
        symmetries += [[3, 2, 0, 1, 6, 7, 5, 4]]
        # Flip x, flip a, flip b in y=0
        symmetries += [[3, 2, 1, 0, 5, 4, 6, 7]]
        # All the above, but swapping A and B
        swapped = [symm[4:] + symm[:4] for symm in symmetries]
        symmetries = symmetries + swapped
        self.assertSetEqual(set(map(tuple, symmetries)),
                            set(map(tuple, self.PRbox_symmetries)),
                            "Failed to discover the symmetries of the PR box.")

    def test_discover_inflation(self):
        GHZ_symmetries = discover_distribution_symmetries(self.GHZ,
                                                          self.triangle)
        # Order: a11=0 a11=1 a12=0 a12=1 b11=0 b11=1 b21=0 b21=1 c11=0 c11=1
        # There's only three symmetries: swap all outcomes, swap A and B, and
        # both at the same time
        # Identity
        symmetries = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        # Flip all outcomes
        symmetries += [[1, 0, 3, 2, 5, 4, 7, 6, 9, 8]]
        # Flip A and B
        symmetries += [[4, 5, 6, 7, 0, 1, 2, 3, 8, 9]]
        # Both at the same time
        symmetries += [[5, 4, 7, 6, 1, 0, 3, 2, 9, 8]]
        self.assertSetEqual(set(map(tuple, symmetries)),
                            set(map(tuple, GHZ_symmetries)),
                            "Failed to discover the symmetries of the PR box.")

    def test_group_elements_from_generators(self):
        generators = [[1, 2, 0], [1, 0, 2]]
        elements = group_elements_from_generators(generators)
        truth = [[0, 1, 2], [1, 0, 2], [2, 1, 0],
                 [2, 0, 1],  [0, 2, 1], [1, 2, 0]]
        self.assertSetEqual(set(map(tuple, elements)),
                            set(map(tuple, truth)),
                            "Failed to generate S3 from generators.")

    def test_desymmetrized_certificate(self):
        self.bellScenario.add_symmetries(self.PRbox_symmetries)
        lp = InflationLP(self.bellScenario, verbose=0)
        lp.set_distribution(self.PR_box)
        lp.solve()
        certificate = lp.desymmetrize_certificate()
        truth = {
            'P[A_0=0]': 0.125, 'P[A_0=1]': 0.125, 'P[A_1=0]': 0.125, 'P[A_1=1]': 0.125,
            'P[B_0=0]': 0.125, 'P[B_0=1]': 0.125, 'P[B_1=0]': 0.125, 'P[B_1=1]': 0.125,
            'P[A_0=0 B_0=0]': -0.1875, 'P[A_0=0 B_0=1]': 0.0625,
            'P[A_0=0 B_1=0]': -0.1875, 'P[A_0=0 B_1=1]': 0.0625,
            'P[A_0=1 B_1=1]': -0.1875, 'P[A_0=1 B_1=0]': 0.0625,
            'P[A_0=1 B_0=1]': -0.1875, 'P[A_0=1 B_0=0]': 0.0625,
            'P[A_1=0 B_0=0]': -0.1875, 'P[A_1=0 B_0=1]': 0.0625,
            'P[A_1=0 B_1=1]': -0.1875, 'P[A_1=0 B_1=0]': 0.0625,
            'P[A_1=1 B_1=0]': -0.1875, 'P[A_1=1 B_1=1]': 0.0625,
            'P[A_1=1 B_0=1]': -0.1875, 'P[A_1=1 B_0=0]': 0.0625}
        self.assertDictEqual(certificate, truth,
                             "Failed to desymmetrize the CHSH inequality.")

