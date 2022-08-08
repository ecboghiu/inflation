from causalinflation import InflationProblem, InflationSDP
from examples_utils import bisection, P_W_array
import numpy as np

qtriangle = InflationProblem(dag={"rho_AB": ["A", "B"],
                                  "rho_BC": ["B", "C"],
                                  "rho_AC": ["A", "C"]}, 
                             outcomes_per_party=[2, 2, 2],
                             settings_per_party=[1, 1, 1],
                             inflation_level_per_source=[2, 2, 2])
sdprelax = InflationSDP(qtriangle, verbose=1)

cols = sdprelax.build_columns('npa2', max_monomial_length=4)
sdprelax.generate_relaxation(cols)

mon1 = sdprelax._monomials_list_all[2:10,1]
mon2 = sdprelax._monomials_list_all[3:5,1]

sdprelax.moment_lowerbounds = {mon: -1 for mon in mon1}
sdprelax.moment_upperbounds = {mon: 3 for mon in mon2}

# v = 0.9
# sdprelax.set_distribution(v*P_W_array+(1-v)*np.ones((2,2,2,1,1,1))/8,  use_lpi_constraints=True)
# sdprelax.solve(dualise=True)
# print(sdprelax.status)

bisection(sdprelax, P_W_array, 1e-6, verbose=True)
