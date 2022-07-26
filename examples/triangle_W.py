from causalinflation import InflationProblem, InflationSDP
from examples_utils import bisection, P_W_array

qtriangle = InflationProblem(dag={"rho_AB": ["A", "B"],
                                  "rho_BC": ["B", "C"],
                                  "rho_AC": ["A", "C"]}, 
                             outcomes_per_party=[2, 2, 2],
                             settings_per_party=[1, 1, 1],
                             inflation_level_per_source=[2, 2, 2])
sdprelax = InflationSDP(qtriangle, verbose=1)

cols = sdprelax.build_columns('physical2', max_monomial_length=4)
sdprelax.generate_relaxation(cols)


# sdprelax.set_distribution(P_W_array)
# sdprelax.solve()

bisection(sdprelax, P_W_array, 1e-4, verbose=True)
