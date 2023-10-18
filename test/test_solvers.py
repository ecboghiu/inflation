import unittest
import numpy as np
import warnings
from scipy.sparse import lil_matrix, coo_matrix, vstack
from copy import deepcopy

from inflation.sdp.sdp_utils import solveSDP_MosekFUSION
from inflation.lp.lp_utils import solveLP_sparse, to_sparse, convert_dicts, \
    solveLP

simple_lp = {
    "objective": {'x': 1, 'y': 1, 'z': 1, 'w': -2},  # x + y + z - 2w
    "known_vars": {'1': 1},  # Define the variable that is the identity
    "inequalities": [{'x': -1, '1': 2},  # 2 - x >= 0
                     {'y': -1, '1': 5},  # 5 - y >= 0
                     {'z': -1, '1': 1 / 2},  # 1/2 - z >= 0
                     {'w': 1, '1': 1}],  # w >= -1
    "equalities": [{'x': 1 / 2, 'y': 2, '1': -3}]  # x/2 + 2y - 3 = 0
}
var = ['1', 'w', 'x', 'y', 'z']
simple_lp_mat = convert_dicts(**simple_lp, variables=var)


class TestSDP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore", category=DeprecationWarning)

    G = np.array([[1,  2,  3,  4,  5],
                  [2,  1,  6,  7,  8],
                  [3,  6,  1,  9, 10],
                  [4,  7,  9,  1, 11],
                  [5,  8, 10, 11,  1]])
    mask_matrices = {}
    for i in np.unique(G):
        mask_matrices.update({str(i): lil_matrix(G == i)})
    simple_sdp = {"mask_matrices": mask_matrices,
                  "objective":  {'7': 1, '8': 1, '9': 1, '10': -1},
                  "known_vars": {'1': 1}
                  }

    def test_LP_with_SDP(self):
        primal_sol   = solveSDP_MosekFUSION(**simple_lp, solve_dual=False)
        dual_sol     = solveSDP_MosekFUSION(**simple_lp, solve_dual=True)
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
                      "equalities": [
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
        for i in range(4):
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


class TestLP(unittest.TestCase):
    lp_expected_sol_free_bounds = {
        "primal_value": 5.5,
        # "dual_certificate": {'1': 5.5, 'w': 2, 'x': -1, 'y': -1, 'z': -1},
        "dual_certificate": {'1': 5.5},
        "x": {'x': 2, 'y': 1, 'z': 1 / 2, 'w': -1, '1': 1}
    }
    lp_expected_sol_non_neg_bounds = {
        "primal_value": 3.5,
        # "dual_certificate": {'1': 3.5, 'w': 2, 'x': -1, 'y': -1, 'z': -1},
        "dual_certificate": {'1': 3.5},
        "x": {'x': 2.0, 'y': 1.0, 'w': 0.0, 'z': 0.5, '1': 1}
    }

    def test_LP_free_bounds(self):
        expected_sol = self.lp_expected_sol_free_bounds
        actual_sols = self.setup_LP_test_case({"default_non_negative": False})
        self.check_solution(expected_sol, actual_sols)

    def test_LP_non_negative_bounds(self):
        expected_sol = self.lp_expected_sol_non_neg_bounds
        actual_sols = self.setup_LP_test_case({"default_non_negative": True})
        self.check_solution(expected_sol, actual_sols)

    def test_LP_lower_bounds_of_zero(self):
        expected_sol = self.lp_expected_sol_non_neg_bounds
        actual_sols = self.setup_LP_test_case({
            "lower_bounds": {'x': 0, 'y': 0, 'z': 0, 'w': 0},
            "default_non_negative": False})
        self.check_solution(expected_sol, actual_sols)

    def test_LP_lower_bounds_of_zero_and_non_negative(self):
        expected_sol = self.lp_expected_sol_non_neg_bounds
        actual_sols = self.setup_LP_test_case({
            "lower_bounds": {'x': 0, 'y': 0, 'z': 0, 'w': 0},
            "default_non_negative": True})
        self.check_solution(expected_sol, actual_sols)

    def test_LP_negative_lower_bounds(self):
        expected_sol = {
            "primal_value": 0.0,
            "dual_certificate": {'1': 8.0},
            "x": {'x': -2.0, 'y': 2.0, 'z': -2.0, 'w': -1.0, '1': 1}}
        actual_sols = self.setup_LP_test_case({
            "lower_bounds": {'x': -2, 'y': 2},
            "upper_bounds": {'z': -2, 'w': 2},
            "default_non_negative": False})
        self.check_solution(expected_sol, actual_sols)

    def test_LP_negative_lower_bounds_and_non_negative(self):
        expected_sol = self.lp_expected_sol_non_neg_bounds
        actual_sols = self.setup_LP_test_case({
            "lower_bounds": {'x': -1, 'y': 1},
            "upper_bounds": {'w': 1},
            "default_non_negative": True})
        self.check_solution(expected_sol, actual_sols)

    @staticmethod
    def setup_LP_test_case(args):
        """Given problem arguments, set up dictionary of solutions from the
        solver."""
        args.update(convert_dicts(**args, variables=var))
        mat_primal_sol = solveLP_sparse(**simple_lp_mat, **args,
                                        variables=var, solve_dual=False)
        mat_dual_sol = solveLP_sparse(**simple_lp_mat, **args,
                                      variables=var, solve_dual=True)
        actual_sols = {
            "sparse, primal": mat_primal_sol,
            "sparse, dual": mat_dual_sol
        }
        return actual_sols

    def check_solution(self, exp_sol, act_sols):
        """Asserts that primal values, certificates, and x values are all
        correct."""
        primal_values = {prob: sol["primal_value"]
                         for prob, sol in act_sols.items()}
        for prob, value in primal_values.items():
            with self.subTest(msg="Testing primal values", i=prob):
                self.assertEqual(exp_sol["primal_value"], value,
                                 f"The objective value is incorrect.")
        certificates = {prob: sol["dual_certificate"]
                        for prob, sol in act_sols.items()}
        for prob, cert in certificates.items():
            with self.subTest(msg="Testing certificates", i=prob):
                self.assertEqual(exp_sol["dual_certificate"], cert,
                                 f"The dual certificate is incorrect.")
        x_values = {prob: sol["x"] for prob, sol in act_sols.items()}
        for prob, x in x_values.items():
            with self.subTest(msg="Testing x values", i=prob):
                self.assertEqual(exp_sol["x"], x,
                                 f"The solution values are incorrect.")


class TestSolverProcesses(unittest.TestCase):
    def test_semiknown_constraints(self):
        """Check that semiknown_moments are correctly processed."""
        problem = {
            "objective":  {'x': -1, 'y': -2, 'z': -1},
            "known_vars": {'1': 1},
            "inequalities":  [{'x': 1, 'y': 1, 'z': 1, '1': -26},
                              {'x': 1, '1': -3},
                              {'y': 1, '1': -4},
                              {'z': 1, '1': -1}],
            "equalities": [{'x': -5, 'y': 1, 'z': -2, '1': -7}]
        }
        p = solveSDP_MosekFUSION(**problem,
                                 semiknown_vars={},
                                 solve_dual=False,
                                 process_constraints=False)
        p_lpi = solveSDP_MosekFUSION(**problem,
                                     semiknown_vars={'z': (0.5, 'x')},
                                     solve_dual=False,
                                     process_constraints=False)
        p_lpi_process = solveSDP_MosekFUSION(**problem,
                                             semiknown_vars={'z': (0.5, 'x')},
                                             solve_dual=False,
                                             process_constraints=True)
        d = solveSDP_MosekFUSION(**problem,
                                 semiknown_vars={},
                                 solve_dual=True,
                                 process_constraints=False)
        d_lpi = solveSDP_MosekFUSION(**problem,
                                     semiknown_vars={'z': (0.5, 'x')},
                                     solve_dual=True,
                                     process_constraints=False)
        d_lpi_process = solveSDP_MosekFUSION(**problem,
                                             semiknown_vars={'z': (0.5, 'x')},
                                             solve_dual=True,
                                             process_constraints=True)
        p_lp = solveLP(**problem, 
                       semiknown_vars={},
                       solve_dual=False)
        p_lpi_lp = solveLP(**problem,
                           semiknown_vars={'z': (0.5, 'x')}, 
                           solve_dual=False)
        d_lp = solveLP(**problem,
                       semiknown_vars={},
                       solve_dual=True)
        d_lpi_lp = solveLP(**problem,
                           semiknown_vars={'z': (0.5, 'x')},
                           solve_dual=True)

        truth_obj, truth_obj_lpi = -52, -109/2
        truth_x =     {'x': 3, 'y': 24, 'z': 1, '1': 1}
        truth_x_lpi = {'x': 3, 'y': 25, 'z': 3/2, '1': 1}

        msg = "The dual and primal solutions are not equal when " + \
              "processing semiknown constraints."

        check = lambda x: np.isclose(x, truth_obj)
        self.assertTrue(all(map(check, [p['primal_value'],
                                        d['primal_value'],
                                        p_lp['primal_value'],
                                        d_lp['primal_value'],
                                        p['dual_value'],
                                        d['dual_value'],
                                        p_lp['dual_value'],
                                        d_lp['dual_value'],
                                        ])), msg)

        check = lambda x: np.isclose(x, truth_obj_lpi)
        self.assertTrue(all(map(check, [p_lpi['primal_value'],
                                        d_lpi['primal_value'],
                                        p_lpi_process['primal_value'],
                                        d_lpi_process['primal_value'],
                                        p_lpi_lp['primal_value'],
                                        d_lpi_lp['primal_value'],
                                        p_lpi['dual_value'],
                                        d_lpi['dual_value'],
                                        p_lpi_process['dual_value'],
                                        d_lpi_process['dual_value'],
                                        p_lpi_lp['dual_value'],
                                        d_lpi_lp['dual_value']
                                        ])), msg)

        check = lambda x: all([np.isclose(v, truth_x[k])
                               for k, v in x.items()])
        self.assertTrue(all(map(check, [p["x"], d["x"],
                                        p_lp["x"], d_lp["x"]])), msg)

        check = lambda x: all([np.isclose(v, truth_x_lpi[k])
                               for k, v in x.items()])
        self.assertTrue(all(map(check, [p_lpi["x"], p_lpi_process['x'],
                                        d_lpi["x"], d_lpi_process['x'],
                                        p_lpi_lp["x"], d_lpi_lp["x"]])), msg)

    def test_partially_known_objective(self):
        """Check that semiknown_moments are correctly processed."""
        problem = {
            "objective":  {'x': -1, 'y': -2, 'z': -1},
            "known_vars": {'1': 1, 'y': 25},
            "inequalities":  [{'x': 1, 'y': 1, 'z': 1, '1': -26},
                              {'x': 1, '1': -3},
                              {'y': 1, '1': -4},
                              {'z': 1, '1': -1}],
            "equalities": [{'x': -5, 'y': 1, 'z': -2, '1': -7}],
            "semiknown_vars": {'z': (0.5, 'x')}
        }
        problem_mat = convert_dicts(**problem, variables=var)
        p_lpi = solveSDP_MosekFUSION(**problem,
                                     solve_dual=False,
                                     process_constraints=False)
        p_lpi_process = solveSDP_MosekFUSION(**problem,
                                             solve_dual=False,
                                             process_constraints=True)
        d_lpi = solveSDP_MosekFUSION(**problem,
                                     solve_dual=True,
                                     process_constraints=False)
        d_lpi_process = solveSDP_MosekFUSION(**problem,
                                             solve_dual=True,
                                             process_constraints=True)
        p_lpi_lp_sparse = solveLP_sparse(**problem_mat,
                                         variables=var,
                                         solve_dual=False)
        d_lpi_lp_sparse = solveLP_sparse(**problem_mat,
                                         variables=var,
                                         solve_dual=True)
        truth_obj_lpi = -109/2
        truth_x_lpi = {'x': 3, 'z': 3/2}

        msg = "The dual and primal solutions are not equal when " + \
              "processing semiknown constraints and partially-known objective."


        check = lambda x: all([np.isclose(v, truth_x_lpi[k])
                               for k, v in truth_x_lpi.items()])
        self.assertTrue(check(p_lpi["x"]), msg)
        self.assertTrue(check(p_lpi_process["x"]), msg)
        self.assertTrue(check(p_lpi_lp_sparse["x"]), msg)
        self.assertTrue(check(d_lpi["x"]), msg)
        self.assertTrue(check(d_lpi_process["x"]), msg)
        self.assertTrue(check(d_lpi_lp_sparse["x"]), msg)

        check = lambda x: np.isclose(x, truth_obj_lpi)

        vals = [
            p_lpi['primal_value'],
            p_lpi['dual_value'],
            p_lpi_process['primal_value'],
            p_lpi_process['dual_value'],
            p_lpi_lp_sparse['primal_value'],
            p_lpi_lp_sparse['dual_value'],
            d_lpi_process['primal_value'],
            d_lpi_process['dual_value'],
            d_lpi['primal_value'],
            d_lpi['dual_value'],
            d_lpi_lp_sparse['primal_value'],
            d_lpi_lp_sparse['dual_value'],
        ]
        self.assertTrue(all(map(check, vals)), msg + "\n" +f"{vals} + vs {truth_obj_lpi}")


class TestTools(unittest.TestCase):
    def test_to_sparse(self):
        with self.subTest(msg="Convert dictionary to sparse matrix"):
            known_vars = {'y': 0, 'x': -2, 'z': 9}
            variables = ['x', 'y', 'z']
            expected_mat = coo_matrix(([-2, 0, 9], ([0, 0, 0], [0, 1, 2])),
                                      shape=(1, 3))
            actual_mat = to_sparse(known_vars, variables)
            self.assertEqual((expected_mat - actual_mat).nnz, 0,
                             "The dictionary was not correctly converted to a "
                             "sparse matrix.")
        with self.subTest(msg="Convert list of dictionaries to sparse matrix"):
            inequalities = simple_lp["inequalities"]
            variables = ['1', 'w', 'x', 'y', 'z']
            expected_mat = coo_matrix(([2, -1, 5, -1, 1/2, -1, 1, 1],
                                       ([0, 0, 1, 1, 2, 2, 3, 3],
                                       [0, 2, 0, 3, 0, 4, 0, 1])),
                                      shape=(4, 5))
            actual_mat = to_sparse(inequalities, variables)
            self.assertEqual((expected_mat - actual_mat).nnz, 0,
                             "The list of dictionaries was not correctly "
                             "converted to a sparse matrix.")

    def test_convert_dicts(self):
        dict_lp = deepcopy(simple_lp)
        semiknown = dict_lp["semiknown_vars"] = {'x': (4, 'y'), 'w': (-2, 'z')}
        dict_lp["lower_bounds"] = {'x': 0, 'y': -2, 'z': 3}
        dict_lp["upper_bounds"] = {'w': 5, 'x': 0, 'y': -1}
        mat_lp = {k: to_sparse(v, var) for k, v in dict_lp.items()
                  if k != "semiknown_vars"}
        semiknown_mat = coo_matrix(([1, -4, 1, 2],
                                    ([0, 0, 1, 1], [2, 3, 1, 4])),
                                   shape=(len(semiknown), len(var)))
        mat_lp["equalities"] = vstack((mat_lp["equalities"], semiknown_mat))
        mixed_lp = {k: dict_lp[k] for k in ("objective", "semiknown_vars")}
        mixed_lp.update({k: mat_lp[k]
                         for k in ("known_vars", "upper_bounds")})

        with self.subTest(msg="Variables not passed"):
            with self.assertRaises(AssertionError):
                convert_dicts(**dict_lp)
        with self.subTest(msg="All arguments are matrices"):
            expected_args = {}
            actual_args = convert_dicts(**mat_lp, variables=var)
            self.assertEqual(expected_args, actual_args)

        types = {"All dictionaries": (mat_lp, dict_lp),
                 "Mixed type": ({"objective": mat_lp["objective"],
                                 "equalities": semiknown_mat},
                                mixed_lp)}
        for type_name, (expected_args, to_convert) in types.items():
            with self.subTest(msg="Convert arguments of type:", i=type_name):
                actual_args = convert_dicts(**to_convert, variables=var)
                self.assertEqual(len(expected_args), len(actual_args),
                                 "The number of returned arguments is not "
                                 "correct.")
                for arg in expected_args:
                    (exp_arg, act_arg) = (expected_args[arg], actual_args[arg])
                    self.assertEqual((exp_arg - act_arg).nnz, 0,
                                     f"{arg} is not equal: "
                                     f"{exp_arg.toarray()} != "
                                     f"{act_arg.toarray()}.")
