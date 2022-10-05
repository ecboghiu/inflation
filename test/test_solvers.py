import unittest
import numpy as np

from causalinflation.quantum.sdp_utils import solveSDP_MosekFUSION

class TestMosek(unittest.TestCase):

    def test_solveSDP_Mosek(self):
        # Test only linear constraints, with no Matrix variables
        solveSDP_arguments = {"objective":  {'x': 1, 'y': 1, 'z': 1, 'w': -2},  # x + y + z - 2w
                              "known_vars": {'1': 1},  # Define the variable that is the identity
                              "var_inequalities":  [{'x': -1, '1': 2},    # 2 - x >= 0
                                                    {'y': -1, '1': 5},    # 5 - y >= 0
                                                    {'z': -1, '1': 1/2},  # 1/2 - z >= 0
                                                    {'w': 1,  '1': 1}],   # w >= -1
                              "var_equalities": [{'x': 1/2, 'y': 2, '1': -3}]  # x/2 + 2y - 3 = 0
        }
        primal_sol   = solveSDP_MosekFUSION(**solveSDP_arguments, solve_dual=False)
        dual_sol     = solveSDP_MosekFUSION(**solveSDP_arguments, solve_dual=True)
        value_primal = primal_sol["primal_value"]
        value_dual   = dual_sol["primal_value"]
        self.assertTrue(np.isclose(value_primal, value_dual), "The dual and primal solutions are not equal.")
        self.assertTrue(np.isclose(value_dual, 2 + 1 + 1/2 + 2), "The solution is not correct.")  # Found with WolframAlpha

        # Test only SDP. Max CHSH by bypassing InflationSDP and setting it by hand
        G = np.array([[1,  2,  3,  4,  5],
                      [2,  1,  6,  7,  8],
                      [3,  6,  1,  9, 10],
                      [4,  7,  9,  1, 11],
                      [5,  8, 10, 11,  1]])
        solveSDP_arguments = {"mask_matrices": {str(i): G == i for i in np.unique(G)},
                              "objective":  {'7': 1, '8': 1, '9': 1, '10': -1},
                              "known_vars": {'1': 1}}
        primal_sol   = solveSDP_MosekFUSION(**solveSDP_arguments, solve_dual=False)
        dual_sol     = solveSDP_MosekFUSION(**solveSDP_arguments, solve_dual=True)
        value_primal = primal_sol["primal_value"]
        value_dual   = dual_sol["primal_value"]
        self.assertTrue(np.isclose(value_primal, value_dual), "The dual and primal solutions are not equal.")
        self.assertTrue(np.abs(value_primal - 2*np.sqrt(2)) < 1e-5, "The solution is not 2√2")
        self.assertTrue(np.abs(dual_sol['x']['7'] - 1/np.sqrt(2)) < 1e-5, "The two body correlator is not 1/√2")

        # Test SDP mixed with inequality constraints. Max CHSH while enforcing CHSH <= 2.23
        solveSDP_arguments = {"mask_matrices": {str(i): G == i for i in np.unique(G)},
                              "objective":  {'7': 1, '8': 1, '9': 1, '10': -1},
                              "known_vars": {'1': 1},
                              "var_inequalities": [{'1': 2.23, '7': -1, '8': -1, '9': -1, '10': 1}]  # CHSH <= 2.23
                            }
        primal_sol   = solveSDP_MosekFUSION(**solveSDP_arguments, solve_dual=False)
        dual_sol     = solveSDP_MosekFUSION(**solveSDP_arguments, solve_dual=True)
        value_primal = primal_sol["primal_value"]
        value_dual   = dual_sol["primal_value"]
        self.assertTrue(np.isclose(value_primal, value_dual), "The dual and primal solutions are not equal.")
        self.assertTrue(np.abs(value_primal - 2.23) < 1e-5, "Max CHSH with CHSH <= 2.23 is not 2.23.")

        # Test SDP mixed with equality constraints plus new variables not
        # in the moment matrix. Max CHSH while enforcing that the 2 body
        # correlators satisfy a local model calculated with Mathematica.
        import sympy as sp
        q = np.zeros((4,4), dtype=object)
        for i in range(4):
            for j in range(4):
                q[i, j] = sp.Symbol(f'q{i}{j}')

        A0B0 =  q[0,0] - q[0,1] + q[0,2] - q[0,3] - q[1,0] + q[1,1] - \
                q[1,2] + q[1,3] + q[2,0] - q[2,1] + q[2,2] - q[2,3] - \
                q[3,0] + q[3,1] - q[3,2] + q[3,3]
        A0B1 =  q[0,0] + q[0,1] - q[0,2] - q[0,3] - q[1,0] - q[1,1] + \
                q[1,2] + q[1,3] + q[2,0] + q[2,1] - q[2,2] - q[2,3] - \
                q[3,0] - q[3,1] + q[3,2] + q[3,3]
        A1B0 =  q[0,0] - q[0,1] + q[0,2] - q[0,3] + q[1,0] - q[1,1] + \
                q[1,2] - q[1,3] - q[2,0] + q[2,1] - q[2,2] + q[2,3] - \
                q[3,0] + q[3,1] - q[3,2] + q[3,3]
        A1B1 =  q[0,0] + q[0,1] - q[0,2] - q[0,3] + q[1,0] + q[1,1] - \
                q[1,2] - q[1,3] - q[2,0] - q[2,1] + q[2,2] + q[2,3] - \
                q[3,0] - q[3,1] + q[3,2] + q[3,3]
        solveSDP_arguments = {"mask_matrices": {str(i): G == i for i in np.unique(G)},
                              "objective":  {'7': 1, '8': 1, '9': 1, '10': -1},
                              "known_vars": {'1': 1},
                              "var_inequalities": [*[{q[i, j]: 1} for i in range(4) for j in range(4)]  # Positivity
                                                   ],
                              "var_equalities": [{**{q[i, j]: 1 for i in range(4) for j in range(4)}, '1': -1},  # Normalisation
                                                 {**A0B0.as_coefficients_dict(), '7': -1},  # LHV
                                                 {**A0B1.as_coefficients_dict(), '8': -1},  # ...
                                                 {**A1B0.as_coefficients_dict(), '9': -1},
                                                 {**A1B1.as_coefficients_dict(), '10': -1}]
                            }
        primal_sol   = solveSDP_MosekFUSION(**solveSDP_arguments, solve_dual=False)
        dual_sol     = solveSDP_MosekFUSION(**solveSDP_arguments, solve_dual=True)
        value_primal = primal_sol["primal_value"]
        value_dual   = dual_sol["primal_value"]
        self.assertTrue(np.isclose(value_primal, value_dual), "The dual and primal solutions are not equal.")
        self.assertTrue(np.abs(value_primal - 2) < 1e-5, "Max CHSH over local strategies is not 2, the local bound.")
        # Check that some of the constraints are satisfied
        vals = dual_sol['x']
        for i in range(4):
            for j in range(4):
                self.assertTrue(vals[q[i, j]] >= -1e-9, f"q[{i}, {j}] is negative.")
        self.assertTrue(np.abs(np.sum([vals[q[i, j]] for i in range(4) for j in range(4)]) - 1) < 1e-9, "q is not normalised.")
