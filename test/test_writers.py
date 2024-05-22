import unittest
import warnings
import mosek
import os
import numpy as np

from inflation import InflationProblem, InflationLP, InflationSDP
from inflation.lp.writer_utils import write_to_lp, write_to_mps
from inflation.sdp.writer_utils import write_to_sdpa


class TestLPWriters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore", category=DeprecationWarning)

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
    sdp.set_extra_inequalities([{"P[A_1=0]": c1, "P[A_1=1]": -c2}])

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
        self.assertTrue(blocks == "4 = number of blocks",
                        "The number of blocks is not correct.")
        self.assertTrue(struct == "(137,-1,-141,-2) = BlockStructure",
                        "The block structure is not correct.")

        # Assert that the file contains the objective function
        objective = contents.split("\n")[4]
        self.assertTrue(objective[0] == '{' and objective[-1] == '}',
                        "The objective function is not recognized.")
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

    def tearDown(self):
        os.remove(f"inst.{self.ext}")
