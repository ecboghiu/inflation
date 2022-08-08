from causalinflation import InflationProblem, InflationSDP
from examples_utils import bisection, P_W_array
import numpy as np

qtriangle = InflationProblem(dag={"rho_AB": ["A", "B"],
                                  "rho_BC": ["B", "C"],
                                  "rho_AC": ["A", "C"]}, 
                             outcomes_per_party=[2, 2, 2],
                             settings_per_party=[2, 2, 2],
                             inflation_level_per_source=[2, 2, 2])
sdprelax = InflationSDP(qtriangle, verbose=1)
sdprelax.generate_relaxation([[],[0],[1],[2],[0,1],[0,2],[1,2]])

mmnts = sdprelax.measurements
A0, B0, C0, A1, B1, C1 = (1-2*mmnts[party][0][setting][0] for setting in range(2) for party in range(3))

sdprelax.set_objective(objective = A1*B0*C0 + A0*B1*C0 + A0*B0*C1 - A1*B1*C1)
print(sdprelax._objective_as_dict)
#sdprelax.solve(dualise=True)

print(sdprelax.objective_value)