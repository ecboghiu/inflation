import unittest
import warnings
import mosek
import os
import numpy as np

from inflation import InflationProblem, InflationLP, InflationSDP
from inflation.lp.writer_utils import write_to_lp, write_to_mps
from inflation.sdp.writer_utils import write_to_csv, write_to_mat, write_to_sdpa
from scipy.io import loadmat


class TestLPWriters(unittest.TestCase):
    p = np.zeros((2, 2, 2, 1))
    p[0, 0, 0, 0] = 0.3
    p[1, 0, 0, 0] = 0.7
    p[0, 0, 1, 0] = 0.7
    p[1, 1, 1, 0] = 0.3
    p = 0.9 * p + 0.1 * (1 / 4)

    instrumental = InflationProblem({"U_AB": ["A", "B"],
                                     "A": ["B"]},
                                    outcomes_per_party=(2, 2),
                                    settings_per_party=(2, 1),
                                    order=("A", "B"),
                                    classical_sources=["U_AB"])
    instrumental_infLP = InflationLP(instrumental,
                                     nonfanout=False,
                                     verbose=False,
                                     include_all_outcomes=True)
    instrumental_infLP.set_distribution(p)
    instrumental_infLP.set_objective(objective={'P[B=1|do(A=0)]': 1},
                                     direction='max')
    instrumental_infLP.solve()
    args = instrumental_infLP._prepare_solver_arguments(separate_bounds=True)
    primal_value = instrumental_infLP.objective_value

    def test_write_to_lp(self):
        self.ext = 'lp'
        write_to_lp(self.args, "inst.lp")
        with mosek.Task() as task:
            task.readdata("inst.lp")
            task.optimize()
            obj = task.getprimalobj(mosek.soltype.bas)
        self.assertAlmostEqual(self.primal_value, obj,
                               msg="The expected value and value when reading "
                                   "from LP are not equal.")

    def test_write_to_mps(self):
        self.ext = 'mps'
        write_to_mps(self.args, "inst.mps")
        with mosek.Task() as task:
            task.readdata("inst.mps")
            task.optimize()
            obj = task.getprimalobj(mosek.soltype.bas)
        self.assertAlmostEqual(self.primal_value, obj,
                               msg="The expected value and value when reading "
                                   "from MPS are not equal.")

    def tearDown(self):
        os.remove(f"inst.{self.ext}")

class TestSDPWriters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore", category=UserWarning)

    # Create an example problem
    scenario = InflationProblem({"Lambda": ["A"]},
                                outcomes_per_party=[3],
                                settings_per_party=[2],
                                inflation_level_per_source=[2])
    sdp = InflationSDP(scenario)
    sdp.generate_relaxation("npa3")
    
    c1 = 0.1
    c2 = -0.2
    v  = 0.5
    ub = 0.4
    lb = 0.3
    sdp.set_objective({"P[A_0=0]": c1, "P[A_1=1]": c2}, "max")
    sdp.set_bounds({"P[A_0=1]": ub})
    sdp.set_bounds({"P[A_0=1]": 0.3}, "lo")
    sdp.set_values({"P[A_1=0]": v}, use_lpi_constraints=True)
    sdp.set_extra_equalities([{"P[A_0=0]": c2, "P[A_0=1]": -c1}])
    sdp.set_extra_inequalities([{"P[A_0=0]": c1, "P[A_0=1]": -c2}])

    def test_write_to_csv(self):
        self.ext = 'csv'
        # Write the problem to a file
        write_to_csv(self.sdp, 'inst.csv')

        # Read the contents of the file
        with open('inst.csv', 'r') as file:
            contents = file.read()

        # Assert that the file contains the objective function
        self.assertIn(f"Objective: {self.c1}*P[A_0=0]{self.c2}*P[A_1=1]",
                      contents,
                      "The objective function is not exported.")

        # Assert that the file contains the variable constraints
        self.assertEqual(float(contents.split("\n")[1].split(",")[3]), self.v,
                        "The variable constraints are not implemented/correct.")
        
        # Assert that the file contains LPI constraints
        self.assertEqual(contents.split("\n")[1].split(",")[13],
                         f"{self.v}*P[A_0=0]",
                         "The LPI constraints are not implemented/correct.")

        # Checks on bounds
        bounds = contents.split("\n")[140:308]
        lbs = list(float(b.split(",")[1]) for b in bounds)
        ubs = [b.split(",")[2] for b in bounds]
        
        # Assert that the file contains the upper bounds
        self.assertIn(str(self.ub), ubs,
                      "The upper bounds are not implemented/correct.")

        # Assert that the file contains the lower bounds
        self.assertTrue(sum(lbs) == self.lb,
                        "The lower bounds are not implemented/correct.")

        # Assert that the file contains the moment equalities
        self.assertIn(f"{self.c2}*P[A_0=0]-{self.c1}*P[A_0=1]", contents,
                      "The moment equalities are not implemented.")
        
        # Assert that the file contains the moment inequalities
        self.assertIn(f"{self.c1}*P[A_0=0]+{abs(self.c2)}*P[A_0=1]", contents,
                      "The moment inequalities are not implemented.")
        
        # ineq = moment_inequalities[0][0][0][0]
        # self.assertTrue(np.array_equal(ineq[0], [[3, 4]])
        #                 and np.array_equal(ineq[1], [[self.c1, -self.c2]]),
        #                 "The moment inequalities are not correct.")
        
    def test_write_to_mat(self):
        self.ext = 'mat'
        # Write the problem to a file
        write_to_mat(self.sdp, 'inst.mat')

        # Read the contents of the file
        contents = loadmat('inst.mat')

        moments_idx2name    = contents['moments_idx2name']
        momentmatrix        = contents['momentmatrix']
        known_moments       = contents['known_moments']
        semiknown_moments   = contents['semiknown_moments']
        objective           = contents['objective']
        moment_lowerbounds  = contents['moment_lowerbounds']
        moment_upperbounds  = contents['moment_upperbounds']
        moment_equalities   = contents['moment_equalities']
        moment_inequalities = contents['moment_inequalities']

        # Assert that the file contains the expected number of variables
        nvars = (len(moments_idx2name) - len(known_moments)
                 - len(semiknown_moments))
        self.assertEqual(nvars, 661,
                         "The number of variables is not correct.")

        # Assert that the file contains the objective function
        self.assertTrue(np.array_equal(objective, [[3, self.c1], [6, self.c2]]),
                        "The objective function is not correct.")

        # Assert that the file contains the variable constraints
        self.assertTrue(np.array_equal(known_moments, [[1, 0],
                                                       [2, 1],
                                                       [5, self.v],
                                                       [18, self.v**2]]),
                        "The variable constraints are not implemented/correct.")
        
        # Assert that the file contains LPI constraints
        self.assertTrue(all(semiknown_moments[:,1] == self.v),
                        "The LPI constraints are not implemented/correct.")

        # Assert that the file contains the upper bounds
        self.assertTrue(np.array_equal(moment_upperbounds, [[4, self.ub]]),
                        "The upper bounds are not implemented/correct.")

        # Assert that the file contains the lower bounds
        self.assertTrue(len(moment_lowerbounds) == 168
                        and np.array_equal(moment_lowerbounds[1], [4, self.lb])
                        and sum(moment_lowerbounds[:, 1]) == self.lb,
                        "The lower bounds are not implemented/correct.")

        # Assert that the file contains the moment equalities
        self.assertEqual(len(moment_equalities), 1,
                         "The moment equalities are not implemented.")
        
        eq = moment_equalities[0][0][0][0]
        self.assertTrue(np.array_equal(eq[0], [[3, 4]])
                        and np.array_equal(eq[1], [[self.c2, -self.c1]]),
                        "The moment equalities are not correct.")

        # Assert that the file contains the moment inequalities
        self.assertEqual(len(moment_inequalities), 1,
                         "The moment inequalities are not implemented.")
        
        ineq = moment_inequalities[0][0][0][0]
        self.assertTrue(np.array_equal(ineq[0], [[3, 4]])
                        and np.array_equal(ineq[1], [[self.c1, -self.c2]]),
                        "The moment inequalities are not correct.")
        
    def test_write_to_sdpa(self):
        self.ext = 'dat-s'
        # Write the problem to a file
        write_to_sdpa(self.sdp, 'inst.dat-s')

        # Read the contents of the file
        with open('inst.dat-s', "r") as file:
            contents = file.read()

        # Assert that the file is not empty
        self.assertNotEqual(contents, "")

        # Assert that the file contains the expected number of variables and blocks
        vars, blocks, struct = contents.split("\n")[1:4]
        self.assertTrue(vars == "661 = number of vars",
                        "The number of variables is not correct.")
        self.assertTrue(blocks == "5 = number of blocks",
                        "The number of blocks is not correct.")
        self.assertTrue(struct == "(137,-1,-141,-2,-1) = BlockStructure",
                        "The block structure is not correct.")

        # Assert that the file contains the objective function
        objective = contents.split("\n")[4]
        self.assertTrue(objective[0] == '{' and objective[-1] == '}',
                        "The objective function is not created.")
        self.assertTrue(float(objective.split(",")[0][1:]) == self.c1
                        and float(objective.split(",")[2]) == self.c2,
                        "The objective function is not correct.")

        # Assert that the file contains the variable constraints
        self.assertIn(f"0\t1\t1\t4\t-{self.v}", contents,
                      "The variable constraints are not implemented/correct.")
        
        # Assert that the file contains LPI constraints
        self.assertIn(f"1\t1\t1\t14\t{self.v}", contents,
                      "The LPI constraints are not implemented/correct.")

        # Assert that the file contains the upper bounds
        self.assertTrue(f"0\t2\t1\t1\t-{self.ub}" in contents
                        and f"2\t2\t1\t1\t-1.0" in contents,
                        "The upper bounds are not implemented/correct.")

        # Assert that the file contains the lower bounds
        self.assertTrue(f"0\t3\t2\t2\t{self.lb}" in contents
                        and f"2\t3\t2\t2\t1.0" in contents,
                        "The lower bounds are not implemented/correct.")

        # Assert that the file contains the moment equalities
        self.assertIn(f"1\t4\t1\t1\t{self.c2}\n"
                      + f"1\t4\t2\t2\t{-self.c2}\n"
                      + f"2\t4\t1\t1\t{-self.c1}\n"
                      + f"2\t4\t2\t2\t{self.c1}", contents,
                      "The moment equalities are not implemented/correct.")

        # Assert that the file contains the moment inequalities
        self.assertIn(f"1\t5\t1\t1\t{self.c1}\n2\t5\t1\t1\t{-self.c2}\n",
                      contents,
                      "The moment inequalities are not implemented/correct.")

    def tearDown(self):
        os.remove(f"inst.{self.ext}")
