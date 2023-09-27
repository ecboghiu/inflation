import unittest
import warnings
import mosek
import os
import numpy as np

from inflation import InflationProblem, InflationLP
from inflation.lp.writer_utils import write_to_lp, write_to_mps


class TestWriters(unittest.TestCase):
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
    instrumental_infLP.set_objective(objective={'<B_1_0_1>': 1},
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
