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
                                    inflation_level_per_source=(1,),
                                    order=("A", "B"),
                                    classical_sources=["U_AB"])
    instrumental_infLP = InflationLP(instrumental,
                                     nonfanout=False,
                                     verbose=False)
    instrumental_infLP.set_distribution(p)
    instrumental_infLP.set_objective(objective={'<B_1_0_0>': 1},
                                     direction='max')
    instrumental_infLP.solve()
    args = instrumental_infLP._prepare_solver_arguments(separate_bounds=True)
    primal_value = instrumental_infLP.objective_value

    def test_writers(self):
        for (write, ext) in ((write_to_lp, 'lp'), (write_to_mps, 'mps')):
            with self.subTest():
                write(self.args, f"inst.{ext}")
                with mosek.Task() as task:
                    task.readdata(f"inst.{ext}")
                    task.optimize()
                    obj = task.getprimalobj(mosek.soltype.bas)
                self.assertAlmostEqual(self.primal_value, obj,
                                       msg=f"The expected value and value when"
                                           f" reading from {ext.upper()} are "
                                           f"not equal.")
                os.remove(f"inst.{ext}")
