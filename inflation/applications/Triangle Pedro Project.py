from inflation import InflationProblem, InflationLP, InflationSDP
import numpy as np

triangle_web = InflationProblem({"lambda": ["a", "b"],
                                 "mu": ["b", "c"],
                                 "sigma": ["a", "c"]},
                                outcomes_per_party=[2, 2, 2],
                                settings_per_party=[1, 1, 1],
                                inflation_level_per_source=[2, 2, 2],
                                order=['a','b','c'],
                                verbose=2)
triangle_lp = InflationLP(triangle_web,
                          nonfanout=False,
                          verbose=2)
print("Extracting factorizing totally unknowable monomials...")
factorizing_unknowable = [m for m in triangle_lp.monomials
    if (m.n_unknowable_factors > 1) and (m.n_knowable_factors == 0)]
for m in factorizing_unknowable:
    print(m)
