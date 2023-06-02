from inflation import InflationProblem, InflationLP
import numpy as np

Inst = InflationProblem({"A": ["B"],
                         "U_AB": ["A", "B"]},
                        outcomes_per_party=(2, 2),
                        settings_per_party=(2, 1),
                        order=("A", "B"))

p_AB_cond_XY = np.zeros((2, 2, 2, 1))
p_AB_cond_XY[0, 0, 0, 0] = 1
p_AB_cond_XY[0, 1, 1, 0] = 1


Inst_LP = InflationLP(Inst, verbose=0)
Inst_LP.generate_lp()
Inst_LP.set_distribution(p_AB_cond_XY)
Inst_LP.solve(dualise=False)
print(Inst_LP.certificate_as_probs())
Inst_LP.solve(dualise=True)
print(Inst_LP.certificate_as_probs())
