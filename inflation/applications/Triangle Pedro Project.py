from inflation import InflationProblem, InflationLP
import numpy as np
from inflation.sdp.optimization_utils import max_within_feasible
from sympy import Symbol

triangle_web = InflationProblem({"lambda": ["a", "b"],
                                 "mu": ["b", "c"],
                                 "sigma": ["a", "c"]},
                                outcomes_per_party=[2, 2, 2],
                                settings_per_party=[1, 1, 1],
                                inflation_level_per_source=[2, 2, 2],
                                order=['a','b','c'],
                                verbose=0)
triangle_lp = InflationLP(triangle_web,
                          nonfanout=False,
                          verbose=0)



def P_GHZ(v=1):
    p = np.zeros((2, 2, 2, 1, 1, 1))
    for a, b, c, x, y, z in np.ndindex(*p.shape):
        if a == b == c:
            p[a, b, c, x, y, z] = 1 / 2
    return v * p + (1 - v) * np.ones(p.shape) / 8


# print("Extracting factorizing totally unknowable monomials...")
# factorizing_unknowable = [m for m in triangle_lp.monomials
#     if (m.n_unknowable_factors > 1) and (m.n_knowable_factors == 0)]
# for m in factorizing_unknowable:
#     print(m)

triangle_lp.set_distribution(P_GHZ(Symbol("v")))
symbolic_moments = triangle_lp.known_moments.copy()
v, cert = max_within_feasible(triangle_lp,
                              symbolic_moments,
                              "dual",
                              return_last_certificate=True)
print("Critical visibility via dual cert:", v, f"\nCertificate:\n{cert}")

triangle_lp.set_distribution(P_GHZ(Symbol("v")))
symbolic_moments = triangle_lp.known_moments.copy()
v, cert = max_within_feasible(triangle_lp,
                              symbolic_moments,
                              "bisection",
                              return_last_certificate=True)
print("Critical visibility via bisect:", v, f"\nCertificate:\n{cert}")
#
# triangle_lp.set_distribution(P_GHZ(0.5 + 0.02))
# triangle_lp.solve(verbose=0)
# print("Status: ", triangle_lp.status)
# triangle_lp.set_distribution(P_GHZ(0.5 - 0.02))
# triangle_lp.solve(verbose=0)
# print("Status: ", triangle_lp.status)
# triangle_lp.set_distribution(P_GHZ(np.sqrt(2)-0.98))
# triangle_lp.solve(verbose=0)
# print("Status: ", triangle_lp.status)
# triangle_lp.set_distribution(P_GHZ(np.sqrt(2)-1.02))
# triangle_lp.solve(verbose=0)
# print("Status: ", triangle_lp.status)
# triangle_lp.set_distribution(P_GHZ(0.99))
# triangle_lp.solve(verbose=0)
# print("Status: ", triangle_lp.status)
# triangle_lp.set_distribution(P_GHZ(1))
# triangle_lp.solve(verbose=0)
# print("Status: ", triangle_lp.status)