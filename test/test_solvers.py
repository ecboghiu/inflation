import unittest
import numpy as np
import warnings


from causalinflation.quantum.sdp_utils import solveSDP_MosekFUSION


class TestMosek(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore", category=DeprecationWarning)

    G = np.array([[1,  2,  3,  4,  5],
                  [2,  1,  6,  7,  8],
                  [3,  6,  1,  9, 10],
                  [4,  7,  9,  1, 11],
                  [5,  8, 10, 11,  1]])
    simple_sdp = {"idx_matrix": G,
                  "idx_dict": {i: str(i) for i in np.unique(G)},
                  "objective":  {'7': 1, '8': 1, '9': 1, '10': -1},
                  "known_vars": {'1': 1}
                  }

    def test_LP(self):
        problem = {
            "objective":  {'x': 1, 'y': 1, 'z': 1, 'w': -2},  # x + y + z - 2w
            "known_vars": {'1': 1},  # Define the variable that is the identity
            "inequalities":  [{'x': -1, '1': 2},    # 2 - x >= 0
                              {'y': -1, '1': 5},    # 5 - y >= 0
                              {'z': -1, '1': 1/2},  # 1/2 - z >= 0
                              {'w': 1,  '1': 1}],   # w >= -1
            "var_equalities": [{'x': 1/2, 'y': 2, '1': -3}]  # x/2 + 2y - 3 = 0
        }
        primal_sol   = solveSDP_MosekFUSION(**problem, solve_dual=False)
        dual_sol     = solveSDP_MosekFUSION(**problem, solve_dual=True)
        value_primal = primal_sol["primal_value"]
        value_dual   = dual_sol["primal_value"]
        self.assertTrue(np.isclose(value_primal, value_dual),
                        "The dual and primal solutions in LP are not equal.")
        self.assertTrue(np.isclose(value_dual, 2 + 1 + 1/2 + 2),
                        "The solution to a simple LP is not correct.")

    def test_SDP(self):
        # Maximization of CHSH on NPA level 1+AB
        primal_sol   = solveSDP_MosekFUSION(**self.simple_sdp,
                                            solve_dual=False)
        dual_sol     = solveSDP_MosekFUSION(**self.simple_sdp, solve_dual=True)
        value_primal = primal_sol["primal_value"]
        value_dual   = dual_sol["primal_value"]
        self.assertTrue(np.isclose(value_primal, value_dual, atol=1e-5),
                        "The dual and primal solutions in SDP are not equal.")
        self.assertTrue(np.isclose(value_primal, 2*np.sqrt(2)),
                        "The solution to a simple SDP is not correct")

    def test_SDP_equalities(self):
        # Maximize CHSH while the two-body correlators satisfy a local model
        import sympy as sp
        q = np.zeros((4, 4), dtype=object)
        for i, j in np.ndindex(4, 4):
            q[i, j] = sp.Symbol(f'q{i}{j}')

        A0B0 = (q[0, 0] - q[0, 1] + q[0, 2] - q[0, 3] - q[1, 0] + q[1, 1] -
                q[1, 2] + q[1, 3] + q[2, 0] - q[2, 1] + q[2, 2] - q[2, 3] -
                q[3, 0] + q[3, 1] - q[3, 2] + q[3, 3])
        A0B1 = (q[0, 0] + q[0, 1] - q[0, 2] - q[0, 3] - q[1, 0] - q[1, 1] +
                q[1, 2] + q[1, 3] + q[2, 0] + q[2, 1] - q[2, 2] - q[2, 3] -
                q[3, 0] - q[3, 1] + q[3, 2] + q[3, 3])
        A1B0 = (q[0, 0] - q[0, 1] + q[0, 2] - q[0, 3] + q[1, 0] - q[1, 1] +
                q[1, 2] - q[1, 3] - q[2, 0] + q[2, 1] - q[2, 2] + q[2, 3] -
                q[3, 0] + q[3, 1] - q[3, 2] + q[3, 3])
        A1B1 = (q[0, 0] + q[0, 1] - q[0, 2] - q[0, 3] + q[1, 0] + q[1, 1] -
                q[1, 2] - q[1, 3] - q[2, 0] - q[2, 1] + q[2, 2] + q[2, 3] -
                q[3, 0] - q[3, 1] + q[3, 2] + q[3, 3])
        problem = {**self.simple_sdp,
                   **{"inequalities": [*[{q[idx]: 1}    # Positivity
                                       for idx in np.ndindex(4, 4)]],
                      "var_equalities": [
                        # Normalization
                        {**{q[idx]: 1 for idx in np.ndindex(4, 4)}, '1': -1},
                        # LHV
                        {**A0B0.as_coefficients_dict(), '7': -1},
                        {**A0B1.as_coefficients_dict(), '8': -1},
                        {**A1B0.as_coefficients_dict(), '9': -1},
                        {**A1B1.as_coefficients_dict(), '10': -1}
                                        ]
                      }
                   }
        primal_sol   = solveSDP_MosekFUSION(**problem, solve_dual=False)
        dual_sol     = solveSDP_MosekFUSION(**problem, solve_dual=True)
        value_primal = primal_sol["primal_value"]
        value_dual   = dual_sol["primal_value"]
        self.assertTrue(np.isclose(value_primal, value_dual),
                        "The dual and primal solutions in SDPs with " +
                        "additional variables are not equal.")
        self.assertTrue(np.isclose(value_primal, 2),
                        "Max CHSH over local strategies is not 2.")
        # Check that some of the constraints are satisfied
        vals = dual_sol['x']
        for idx in range(4):
            for j in range(4):
                self.assertTrue(vals[q[i, j]] >= -1e-9,
                                f"q[{i}, {j}] is negative.")
        self.assertTrue(np.isclose(1,
                                   sum([vals[q[idx]]
                                        for idx in np.ndindex(4, 4)])),
                        "The local model is not normalized.")

    def test_SDP_inequalities(self):
        # Maximize CHSH on NPA level 1+AB subject to CHSH <= bound
        bound = 2.23
        problem = {**self.simple_sdp,
                   **{"inequalities":
                      [{'1': bound, '7': -1, '8': -1, '9': -1, '10': 1}]}
                   }
        primal_sol   = solveSDP_MosekFUSION(**problem, solve_dual=False)
        dual_sol     = solveSDP_MosekFUSION(**problem, solve_dual=True)
        value_primal = primal_sol["primal_value"]
        value_dual   = dual_sol["primal_value"]
        self.assertTrue(np.isclose(value_primal, value_dual),
                        "The dual and primal solutions in bounded SDPs are " +
                        "not equal.")
        self.assertTrue(np.isclose(value_primal, bound),
                        f"Max CHSH with CHSH <= {bound} is not {bound}.")
