import mosek
from inflation import InflationProblem, InflationLP
import numpy as np

p_ABC = np.zeros((3, 2, 2, 1, 1, 1), dtype=float)
p_ABC[1, 0, 0] = p_ABC[2, 1, 0] = p_ABC[2, 1, 1] = 1/3

Three_Way = InflationProblem({"U_ABC": ["A", "B", "C"],
                              "A": ["B"],
                              "B": ["C"]},
                             outcomes_per_party=(3, 2, 2),
                             settings_per_party=(1, 1, 1),
                             inflation_level_per_source=(1,),
                             order=("A", "B", "C"))

# Incompatible
Three_Way_Unpacked = InflationLP(Three_Way,
                                 nonfanout=False,
                                 include_all_outcomes=True,
                                 verbose=2)
semiknown_usage = False
Three_Way_Unpacked.set_distribution(p_ABC, use_lpi_constraints=semiknown_usage)
Three_Way_Unpacked.update_values({
    "<B_1_0_0 C_1_0_0>": 1/3,
    "<B_1_1_0 C_1_0_0>": 1/3,
    "<B_1_1_0 C_1_0_1>": 1/3,
    "<B_1_1_1 C_1_1_1>": 1/3,
    "<B_1_2_1 C_1_1_0>": 1/3,
    "<B_1_0_1 C_1_1_1>": 2/3,
    "<B_1_2_1 C_1_1_1>": 2/3
},
    use_lpi_constraints=semiknown_usage)

params = {mosek.iparam.optimizer: mosek.optimizertype.dual_simplex}
Three_Way_Unpacked.solve(dualise=False, verbose=2, solverparameters=params)
print("Status: ", Three_Way_Unpacked.status)

# Compatible
Three_Way_Interrupted = InflationLP(Three_Way,
                                    nonfanout=True,
                                    include_all_outcomes=True,
                                    verbose=2)
print(Three_Way_Interrupted.monomial_names)
Three_Way_Interrupted.set_distribution(p_ABC,
                                       use_lpi_constraints=semiknown_usage)
Three_Way_Interrupted.update_values({
    "<B_1_0_0 C_1_0_0>": 1/3,
    "<B_1_1_0 C_1_0_0>": 1/3,
    "<B_1_1_0 C_1_0_1>": 1/3,
    "<B_1_1_1 C_1_1_1>": 1/3,
    "<B_1_2_1 C_1_1_0>": 1/3,
    "<B_1_0_1 C_1_1_1>": 2/3,
    "<B_1_2_1 C_1_1_1>": 2/3
},
    use_lpi_constraints=semiknown_usage)
Three_Way_Interrupted.solve(dualise=False, verbose=2, solverparameters=params)
print("Status: ", Three_Way_Interrupted.status)
