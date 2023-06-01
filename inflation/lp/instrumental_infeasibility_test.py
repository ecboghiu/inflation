from inflation import InflationProblem, InflationLP
import numpy as np

Inst = InflationProblem({"A": ["B"],
                         "U_AB": ["A", "B"]},
                        outcomes_per_party=(2, 2),
                        settings_per_party=(2, 1),
                        order=("A", "B"))

p_X = [0.5, 0.5]
p_AB_cond_X = np.zeros((2, 2, 2))
p_AB_cond_X[0, 0, 0] = 1
p_AB_cond_X[0, 1, 1] = 1

p_ABX = np.zeros((2, 2, 2, 1))
for (a, b, x), v in np.ndenumerate(p_AB_cond_X):
    p_ABX[a, b, x] = v * p_X[x]

Inst_LP = InflationLP(Inst, verbose=2)
Inst_LP.generate_lp()
Inst_LP.set_distribution(p_ABX)
Inst_LP.solve(dualise=False)
Inst_LP.solve(dualise=True)
