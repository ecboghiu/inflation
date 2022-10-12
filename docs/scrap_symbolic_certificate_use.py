from causalinflation import InflationProblem, InflationSDP
from causalinflation.quantum.optimization_utils import maximize_scalar_such_that_sdp_feasible_using_dual_solution, maximize_scalar_such_that_sdp_feasible_using_bisection
import numpy as np

qbilocal = InflationProblem(dag={'rho_AB': ['A', 'B'],
                                 'rho_BC': ['B', 'C']},
                            outcomes_per_party=(2, 2, 2),
                            settings_per_party=(2, 2, 2),
                            inflation_level_per_source=(2, 2))
sdp = InflationSDP(qbilocal)
sdp.generate_relaxation('npa2')


def P_2PR(vis=1) -> np.ndarray:
    p = np.zeros((2, 2, 2, 2, 2, 2))
    for a, b, c, x, y, z in np.ndindex(*p.shape):
        p[a, b, c, x, y, z] = (1 + (-1) ** (a + b + c + x*y + y*z)) / 8
    return vis * p + (1 - vis) * np.ones(p.shape) / 2 ** 3

print(maximize_scalar_such_that_sdp_feasible_using_dual_solution(sdp=sdp,
                                                                 distribution_as_scalar_function=P_2PR,
                                                                 verbose=True))

print(maximize_scalar_such_that_sdp_feasible_using_bisection(sdp=sdp,
                                                             distribution_as_scalar_function=P_2PR,
                                                             verbose=True))